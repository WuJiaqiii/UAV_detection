#!/usr/bin/env python3
"""Convert fast-inference CSVs to competition XLSX files (Python standard library only)."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import csv
import json
from pathlib import Path, PurePosixPath
import re
import sys
import hashlib
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
from util.competition_protocol import validate_counts
from util.burst_normalization import normalize_bursts


# Official supplied workbook: model name -> submission ID, never checkpoint IDs.
KNOWN = dict(zip([
    "DAUTEL EVO NANO", "DEVENTION DEVO", "DJFPV", "DJI AVATA2", "DJI FPV COMBO", "DJI MAVIC3 PRO",
    "DJI MINI3 PRO", "DJI_Inspire_2", "DJI_MATRICE_200", "DJI_MATRICE_600_Pro", "DJMavicMini", "DJMiniSE",
    "EVOIIPRO", "FIMIx8Mini", "FIMIX8SE2020", "FLYSKY FS I6X", "FRSKY X14", "FRSKY X20R",
    "HubsanZino2", "JUMPER T14"], range(20)))
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
FILE_ID = re.compile(r"((1-[1-4]|2-[12])_\d{4})\.dat", re.I)
UNKNOWN = re.compile(r"Unknown_(\d+)", re.I)
UNRESOLVED = {"unknown_unassigned", "uncertain", "no_signal", "error", ""}


def normalized(text):
    return " ".join(str(text).strip().casefold().split())


def identify(path):
    name = PurePosixPath(str(path).replace("\\", "/")).name
    match = FILE_ID.fullmatch(name)
    if not match:
        raise ValueError(f"Expected a competition DAT filename, got {name!r}")
    return match[1], match[2]


def read_xlsx_rows(path):
    """Read the first worksheet; sufficient for the supplied two-column label map."""
    tag = lambda name: f"{{{NS}}}{name}"
    with ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(t.text or "" for t in item.iter(tag("t")))
                      for item in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(z.read("xl/workbook.xml")).find(tag("sheets"))[0]
        rid = sheet.attrib[f"{{{REL}}}id"]
        links = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(x.attrib["Target"] for x in links if x.attrib["Id"] == rid)
        target = target.lstrip("/") if target.startswith("/") else "xl/"+target
        rows = []
        for row in ET.fromstring(z.read(target)).find(tag("sheetData")):
            values = {}
            for cell in row:
                if cell.find(tag("f")) is not None:
                    raise ValueError("Label mapping must contain values, not formulas")
                ref = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                value = cell.findtext(tag("v"))
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(tag("t")))
                values[ref] = value
            rows.append([values.get("A"), values.get("B")])
        return rows


def label_mapping(path=None, aliases=None):
    mapping = KNOWN.copy()
    if path:
        if path.suffix.lower() == ".xlsx":
            mapping = {}
            for name, code in read_xlsx_rows(path)[1:]:
                if name is None and code is None:
                    continue
                if name is None or code is None:
                    raise ValueError("Incomplete label mapping row")
                if name in mapping:
                    raise ValueError(f"Duplicate mapping name: {name}")
                mapping[name] = code
        else:
            mapping = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {}
    for name, code in mapping.items():
        number = Decimal(str(code))
        if isinstance(code, bool) or not number.is_finite() or number != int(number):
            raise ValueError(f"Non-integer class ID for {name}")
        key = normalized(name)
        if not key or key in result:
            raise ValueError(f"Duplicate/empty normalized class name: {name}")
        result[key] = int(number)
    if len(result) != 20 or set(result.values()) != set(range(20)):
        raise ValueError("Official known mapping must contain exactly the 20 distinct IDs 0..19")
    if aliases:
        for alias, name in json.loads(aliases.read_text(encoding="utf-8-sig")).items():
            key = normalized(alias)
            if normalized(name) not in result or key in result:
                raise ValueError(f"Invalid/colliding alias: {alias}")
            result[key] = result[normalized(name)]
    return result


def write_xlsx(path, headers, rows):
    """Write typed numeric IDs/Hz and literal file identifiers, with one Sheet1."""
    q = lambda name: f"{{{NS}}}{name}"
    ET.register_namespace("", NS)
    sheet = ET.Element(q("worksheet"))
    views = ET.SubElement(sheet, q("sheetViews"))
    view = ET.SubElement(views, q("sheetView"), workbookViewId="0")
    ET.SubElement(view, q("pane"), ySplit="1", topLeftCell="A2", state="frozen", activePane="bottomLeft")
    cols = ET.SubElement(sheet, q("cols"))
    for i in range(len(headers)):
        ET.SubElement(cols, q("col"), min=str(i+1), max=str(i+1), width="28" if i == 0 else "22", customWidth="1")
    data = ET.SubElement(sheet, q("sheetData"))
    for i, values in enumerate([headers, *rows], 1):
        row = ET.SubElement(data, q("row"), r=str(i))
        for j, value in enumerate(values):
            cell = ET.SubElement(row, q("c"), r=f"{chr(65+j)}{i}", s="1" if i == 1 else "0")
            if isinstance(value, int):
                ET.SubElement(cell, q("v")).text = str(value)
            else:
                cell.set("t", "inlineStr")
                ET.SubElement(ET.SubElement(cell, q("is")), q("t")).text = str(value)
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        "_rels/.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="'+REL+'/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": '<workbook xmlns="'+NS+'" xmlns:r="'+REL+'"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="'+REL+'/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="'+REL+'/styles" Target="styles.xml"/></Relationships>',
        "xl/styles.xml": '<styleSheet xmlns="'+NS+'"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
        "xl/worksheets/sheet1.xml": ET.tostring(sheet, encoding="unicode"),
    }
    with ZipFile(path, "x", ZIP_DEFLATED) as z:
        for name, content in parts.items():
            z.writestr(name, '<?xml version="1.0" encoding="UTF-8"?>\n'+content)


def frequency(low, high, policy):
    try:
        values = [Decimal(str(x)) * 1000000 for x in (low, high)]
        if not all(x.is_finite() for x in values):
            raise ValueError("non-finite frequency")
        lo, hi = [int(x.quantize(Decimal(1), rounding=ROUND_HALF_UP)) for x in values]
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValueError("Missing or invalid relative frequency in MHz") from exc
    if lo >= hi:
        raise ValueError(f"Frequency lower bound must be below upper bound: {lo}, {hi}")
    clipped = lo < -50000000 or hi > 50000000
    if clipped and policy == "error":
        raise ValueError(f"Frequency outside -50..+50 MHz: {lo}, {hi} Hz; use --frequency-policy clip to intersect")
    lo, hi = max(-50000000, lo), min(50000000, hi)
    if lo >= hi:
        return None, None, True  # the explicit clip policy has an empty intersection
    return lo, hi, clipped


def load_details(paths, wanted):
    bursts = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample, _ = identify(row["path"])
                if sample not in wanted:
                    continue
                for group in row.get("groups", []):
                    key = (sample, group["track_id"])
                    for pulse in group.get("pulses", []):
                        bursts[key].append(pulse)
    return bursts


def build_plan(args):
    csv.field_size_limit(16 * 1024 * 1024)  # compact burst lists can exceed csv's 128 KiB default
    mapping = label_mapping(args.label_map, args.aliases)
    report = {"errors": [], "warnings": [], "datasets": {}, "unknown_labels": {},
              "task2_level": args.task2_level, "unresolved_policy": args.unresolved,
              "frequency_policy": args.frequency_policy, "clipped_intervals": 0,
              "dropped_outside_band": 0, "unresolved_items": [],'task2_row_errors':0}
    error = report["errors"].append
    records, seen, owners, details = [], set(), {}, set(args.details or [])
    for source in args.predictions:
        source_key = source.resolve()
        if not args.details and (source.parent/"slices.jsonl").is_file():
            details.add(source.parent/"slices.jsonl")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if len(reader.fieldnames or [])!=len(set(reader.fieldnames or [])):
                raise ValueError('Duplicate CSV headers')
            detailed='single_signal_prediction' in (reader.fieldnames or [])
            if not {'path'} <= set(reader.fieldnames or []) or (not detailed and not {"label", "status"} <= set(reader.fieldnames or [])):
                raise ValueError(f"{source}: expected path,label,status columns")
            for line, row in enumerate(reader, 2):
                try:
                    if None in row or any(v is None for v in row.values()):raise ValueError('Malformed CSV row width')
                    if detailed:
                        decision=json.loads(row['single_signal_prediction'])
                        if not isinstance(decision,dict):raise ValueError('Missing single-signal decision')
                        if int(row.get('error_slices') or 0)>0 and not args.allow_partial_errors:raise ValueError('Upstream slice errors')
                        row.update(label=decision.get('display_label',''),status=decision.get('recognition_status',''))
                    sample, dataset = identify(row["path"])
                    owner = owners.setdefault(dataset, source_key)
                    if owner != source_key:
                        raise ValueError(f"{dataset} occurs in multiple CSVs: cluster numbers are batch-local; run inference for the whole test set")
                    key = (sample, row.get("track_id", "")) if dataset.startswith("2") else (sample, "")
                    if key in seen:
                        raise ValueError(f"Duplicate file/track: {key}")
                    seen.add(key)
                    records.append({**row, "sample": sample, "dataset": dataset, "where": f"{source.name}:{line}"})
                except (ValueError,TypeError) as exc:
                    error(f"{source.name}:{line}: {exc}")
    if not records:
        error("No prediction rows")
    wanted = {r["sample"] for r in records if r["dataset"].startswith("2")}
    burst_map = load_details(sorted(details), wanted) if args.task2_level == "burst" else {}
    files = defaultdict(list)
    event_files = defaultdict(list)
    class_keys = defaultdict(set)
    for row in records:
        sample, dataset = row["sample"], row["dataset"]
        files.setdefault((dataset, sample), [])
        try:
            if row.get("error", "").strip() and not args.allow_partial_errors:
                raise ValueError("Inference error present; review or use --allow-partial-errors")
            if dataset.startswith("2") and row["status"] == "no_signal":
                if (row.get('label', '').strip().lower() != 'no_signal' or row.get('track_id', '').strip() or
                        row.get('bursts_json', '').strip() not in ('', '[]')):
                    raise ValueError('no_signal must have label no_signal and no track/bursts')
                files[(dataset, sample)] = [[-1, 0, 0]]  # official example: interference-only scene
                continue
            label = row["label"].strip()
            match = UNKNOWN.fullmatch(label)
            if normalized(label) in mapping and row["status"] == "known":
                cls = mapping[normalized(label)]
            elif match and int(match[1]) > 0 and row["status"] == "novel_candidate":
                cls = f"Unknown_{int(match[1]):06d}"
            elif normalized(label) in UNRESOLVED:
                report["unresolved_items"].append({"sample": sample, "track_id": row.get("track_id", ""), "label": label})
                if args.unresolved == "error":
                    raise ValueError(f"Unresolved prediction {label!r}; choose --unresolved singleton/shared explicitly or correct inference")
                cls = "~unresolved" if args.unresolved == "shared" else f"~unresolved:{sample}:{row.get('track_id', '')}"
            else:
                raise ValueError(f"Unmapped or inconsistent label/status: {label!r}/{row['status']}")
            if isinstance(cls, str):
                class_keys[dataset].add(cls)
            if dataset.startswith("1"):
                files[(dataset, sample)].append([cls])
                continue
            if not row.get("track_id"):
                raise ValueError("Task 2 requires the multi-mode CSV with track_id")
            if args.task2_level == "track":
                pulses = [row]
            elif row.get("bursts_json", "").strip():
                pulses = json.loads(row["bursts_json"])
            else:
                pulses = burst_map.get((sample, row["track_id"]), [])
            if not isinstance(pulses, list) or not pulses:
                raise ValueError("Burst data missing: provide --details slices.jsonl, rerun updated fast mode, or explicitly use --task2-level track (envelopes only)")
            if args.task2_level=='track':
                normalized_events,diagnostics=pulses,{'track_envelope_preview':True}
            else:
                normalized_events,diagnostics=normalize_bursts([{**p,'track_id':row.get('track_id'),'label':str(cls),'recognition_status':row['status']} for p in pulses],sample)
            report.setdefault('burst_normalization',[]).append({'sample':sample,'track_id':row.get('track_id'),**diagnostics})
            for pulse in normalized_events:
                if args.task2_level=='burst' and pulse['end_ms']>100.000001:raise ValueError('Burst end exceeds the 100ms recording')
                lo, hi, clipped = frequency(pulse.get("frequency_low_offset_mhz"), pulse.get("frequency_high_offset_mhz"), args.frequency_policy)
                report["clipped_intervals"] += clipped
                if lo is None:
                    report["dropped_outside_band"] += 1
                    continue
                files[(dataset, sample)].append([cls, lo, hi])
                if args.task2_level=='burst':
                    event_files[(dataset,sample)].append({**pulse,'label':cls,
                        'frequency_low_offset_mhz':lo/1e6,'frequency_high_offset_mhz':hi/1e6})
        except (ValueError, TypeError, AttributeError) as exc:
            if dataset.startswith('2'):report['task2_row_errors']+=1
            error(f"{row['where']} {sample}/{row.get('track_id', '')}: {exc}")
    for key,events in event_files.items():
        events,diagnostics=normalize_bursts(events,key[1])
        if diagnostics['label_conflicts']:error(f'{key[1]}: conflicting labels on duplicate burst observations')
        files[key]=[[e['label'],int(round(e['frequency_low_offset_mhz']*1e6)),int(round(e['frequency_high_offset_mhz']*1e6))] for e in events]
    statuses=defaultdict(set)
    for row in records:statuses[row['sample']].add(row['status'])
    for sample,values in statuses.items():
        if 'no_signal' in values and len(values)>1:error(f'{sample}: no_signal conflicts with detections')
    source_coverage = True
    if args.input_root:
        expected_list = []
        for root in args.input_root:
            if not root.is_dir():
                error(f'Source DAT directory does not exist: {root}');source_coverage = False
                continue
            expected_list.extend(identify(p.name)[0] for p in root.rglob('*') if p.is_file() and p.suffix.lower()=='.dat')
        expected = set(expected_list)
        if len(expected) != len(expected_list):
            error('Duplicate source DAT identifiers across input roots');source_coverage = False
        if not expected:
            error('No source DAT files found');source_coverage = False
        actual = {r["sample"] for r in records}
        if expected != actual:
            source_coverage = False
            error(f"CSV/source coverage mismatch: missing={sorted(expected-actual)[:20]}, extra={sorted(actual-expected)[:20]}")
    for dataset in sorted({r["dataset"] for r in records}):
        rows = [r for r in records if r["dataset"] == dataset]
        codes = {key: i+20 for i, key in enumerate(sorted(class_keys[dataset]))}
        report["unknown_labels"][dataset] = codes
        report["datasets"][dataset] = {"files": len({r["sample"] for r in rows}), "input_rows": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "output_signal_rows": sum(sum(v[0] != -1 for v in values) for (ds, _), values in files.items() if ds == dataset),
            "no_signal_files": sum(ds == dataset and values == [[-1, 0, 0]] for (ds, _), values in files.items())}
        if dataset.startswith("2") and report["datasets"][dataset]["files"] != 1750:
            report["warnings"].append(f"{dataset}: {report['datasets'][dataset]['files']} files; the specification describes 1750. Use --input-root to check coverage.")
        for (ds, sample), values in files.items():
            if ds == dataset:
                for value in values:
                    if isinstance(value[0], str):
                        value[0] = codes[value[0]]
    if args.task2_level == "track" and wanted:
        report["warnings"].append("Task 2 uses track envelopes; repeated bursts and hopping sub-bands cannot be recovered. This does not fully implement the burst-list requirement.")
    if report["unresolved_items"] and args.unresolved != "error":
        report["warnings"].append("Unresolved predictions were explicitly assigned artificial unknown IDs; this is a fallback, not inferred class evidence.")
    counts={item:len({r[0] for (ds,_),rows in files.items() if ds==item for r in rows if isinstance(r[0],int) and r[0]>=20}) for item in report['datasets']}
    report['class_budget_errors']=validate_counts(counts)
    for (dataset,sample),values in files.items():
        if dataset.startswith('2') and len({r[0] for r in values})>3:
            report['class_budget_errors'].append(f'{sample}: more than three distinct classes')
    coverage=source_coverage
    for dataset,info in report['datasets'].items():
        ids=sorted(int(r['sample'].split('_')[1]) for r in records if r['dataset']==dataset)
        ids=sorted(set(ids))
        if ids!=list(range(len(ids))):coverage=False
        expected=args.expected_count if args.expected_count is not None else (1750 if dataset.startswith('2') else None)
        if expected is not None and len(ids)!=expected:coverage=False
        if dataset.startswith('2') and len(ids)!=1750:coverage=False
        if expected is None and not args.input_root:coverage=False
    report.update(coverage_complete=coverage,class_budget_valid=not report['class_budget_errors'],
        has_fallbacks=bool(report['unresolved_items']),frequency_valid=not (report['clipped_intervals'] or report['dropped_outside_band'] or report['task2_row_errors']),
        upstream_error_rows=sum(bool(r.get('error','').strip()) or r.get('status')=='error' or int(r.get('error_slices') or 0)>0 for r in records))
    return files, report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, nargs="+", required=True, help="One inference CSV per test set")
    p.add_argument("--output", type=Path, required=True, help="New or empty directory; only submission XLSX files go here")
    p.add_argument("--label-map", type=Path, help="Official XLSX or JSON; default is the supplied official 20-class mapping")
    p.add_argument("--aliases", type=Path, help="JSON of model-name alias -> official model name")
    p.add_argument("--details", type=Path, nargs="+", help="Task 2 slices.jsonl; otherwise automatically look beside each CSV")
    p.add_argument("--task2-level", choices=["burst", "track"], default="burst")
    p.add_argument("--unresolved", choices=["error", "singleton", "shared"], default="error",
                   help="Default error; artificial unknown IDs are allowed only in explicit previews")
    p.add_argument("--frequency-policy", choices=["error", "clip"], default="error")
    p.add_argument("--allow-partial-errors", action="store_true")
    p.add_argument("--input-root", type=Path, nargs="+", help="Optional local DAT roots for missing/extra file checks")
    p.add_argument("--check", action="store_true", help="Validate and write only the sidecar report, without XLSX files")
    p.add_argument('--preview',action='store_true',help='Allow non-ready previews, always flagged in the report')
    p.add_argument('--expected-count',type=int,help='Expected files per test item; task 2 defaults to 1750')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.expected_count is not None and args.expected_count<1:raise ValueError('expected-count must be positive')
    output = args.output.expanduser().resolve()
    report_path = output.parent/(output.name+"_report.json")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")
    files, report = build_plan(args)
    burst_complete = (args.task2_level == "burst" and not report['task2_row_errors']) or not any(ds.startswith("2") for ds, _ in files)
    ready=not report['errors'] and burst_complete and report['coverage_complete'] and report['class_budget_valid'] and not report['has_fallbacks'] and report['frequency_valid'] and not report['upstream_error_rows']
    report.update(output=str(output), report=str(report_path), valid_format=not report["errors"],
                  burst_complete=burst_complete, ready=ready,preview=args.preview,written_workbooks=0)
    if not args.check and report["valid_format"] and (ready or args.preview):
        output.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.submission-',dir=output.parent))
        task1 = defaultdict(list)
        for (dataset, sample), rows in sorted(files.items()):
            if dataset.startswith("1"):
                task1[dataset].append([sample, rows[0][0]])
            else:
                folder = staging/dataset
                folder.mkdir(exist_ok=True)
                write_xlsx(folder/(sample+".xlsx"), ["无人机类别", "起始频率(Hz)", "终止频率(Hz)"], rows)
                report["written_workbooks"] += 1
        for dataset, rows in task1.items():
            write_xlsx(staging/(dataset+".xlsx"), ["序号", "无人机类别"], rows)
            report["written_workbooks"] += 1
        # Independent OOXML read-back: typed numeric values, literal IDs, headers/counts.
        for pth in staging.rglob('*.xlsx'):
            with ZipFile(pth) as archive:
                root=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
                data=root.find(f'{{{NS}}}sheetData')
                assert data is not None and len(data)>=1
                for row in list(data)[1:]:
                    numeric=list(row)[1:] if pth.stem.startswith('1-') else list(row)
                    if any(c.get('t') is not None or c.find(f'{{{NS}}}v') is None for c in numeric):
                        raise ValueError(f'Numeric cell verification failed: {pth.name}')
                actual=[]
                for row in data:
                    actual.append([''.join(t.text or '' for t in c.iter(f'{{{NS}}}t')) if c.get('t')=='inlineStr' else int(c.findtext(f'{{{NS}}}v')) for c in row])
                headers=['序号','无人机类别'] if pth.stem.startswith('1-') else ['无人机类别','起始频率(Hz)','终止频率(Hz)']
                expected=task1[pth.stem] if pth.stem.startswith('1-') else files[(pth.parent.name,pth.stem)]
                if actual != [headers,*expected]:raise ValueError(f'Workbook round-trip mismatch: {pth.name}')
        report['xlsx_sha256']={p.relative_to(staging).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.rglob('*.xlsx')}
        if output.exists():output.rmdir()  # only the pre-checked empty directory
        staging.rename(output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid_format", "burst_complete", "ready",'coverage_complete','class_budget_valid','class_budget_errors','has_fallbacks','frequency_valid','upstream_error_rows', "datasets", "written_workbooks", "warnings", "report")}, ensure_ascii=False, indent=2))
    for message in report["errors"][:8]:
        print("ERROR: "+message)
    return 2 if report['errors'] or (not ready and not args.preview) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, InvalidOperation, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
