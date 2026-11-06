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
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


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
              "dropped_outside_band": 0, "unresolved_items": []}
    error = report["errors"].append
    records, seen, owners, details = [], set(), {}, set(args.details or [])
    for source in args.predictions:
        source_key = source.resolve()
        if not args.details and (source.parent/"slices.jsonl").is_file():
            details.add(source.parent/"slices.jsonl")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "label", "status"} <= set(reader.fieldnames or []):
                raise ValueError(f"{source}: expected path,label,status columns")
            for line, row in enumerate(reader, 2):
                try:
                    sample, dataset = identify(row["path"])
                    owner = owners.setdefault(dataset, source_key)
                    if owner != source_key:
                        raise ValueError(f"{dataset} occurs in multiple CSVs: cluster numbers are batch-local; run inference for the whole test set")
                    key = (sample, row.get("track_id", "")) if dataset.startswith("2") else (sample, "")
                    if key in seen:
                        raise ValueError(f"Duplicate file/track: {key}")
                    seen.add(key)
                    records.append({**row, "sample": sample, "dataset": dataset, "where": f"{source.name}:{line}"})
                except ValueError as exc:
                    error(f"{source.name}:{line}: {exc}")
    if not records:
        error("No prediction rows")
    wanted = {r["sample"] for r in records if r["dataset"].startswith("2")}
    burst_map = load_details(sorted(details), wanted) if args.task2_level == "burst" else {}
    files = defaultdict(list)
    class_keys = defaultdict(set)
    for row in records:
        sample, dataset = row["sample"], row["dataset"]
        files.setdefault((dataset, sample), [])
        try:
            if row.get("error", "").strip() and not args.allow_partial_errors:
                raise ValueError("Inference error present; review or use --allow-partial-errors")
            if dataset.startswith("2") and row["status"] == "no_signal":
                continue  # header-only workbook truthfully reports no detections
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
            for pulse in pulses:
                lo, hi, clipped = frequency(pulse.get("frequency_low_offset_mhz"), pulse.get("frequency_high_offset_mhz"), args.frequency_policy)
                report["clipped_intervals"] += clipped
                if lo is None:
                    report["dropped_outside_band"] += 1
                    continue
                files[(dataset, sample)].append([cls, lo, hi])
        except (ValueError, TypeError, AttributeError) as exc:
            error(f"{row['where']} {sample}/{row.get('track_id', '')}: {exc}")
    if args.input_root:
        expected = {identify(p.name)[0] for root in args.input_root for p in root.rglob("*.dat")}
        actual = {r["sample"] for r in records}
        if expected != actual:
            error(f"CSV/source coverage mismatch: missing={sorted(expected-actual)[:20]}, extra={sorted(actual-expected)[:20]}")
    for dataset in sorted({r["dataset"] for r in records}):
        rows = [r for r in records if r["dataset"] == dataset]
        codes = {key: i+20 for i, key in enumerate(sorted(class_keys[dataset]))}
        report["unknown_labels"][dataset] = codes
        report["datasets"][dataset] = {"files": len({r["sample"] for r in rows}), "input_rows": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "output_signal_rows": sum(len(values) for (ds, _), values in files.items() if ds == dataset)}
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
    return files, report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, nargs="+", required=True, help="One inference CSV per test set")
    p.add_argument("--output", type=Path, required=True, help="New or empty directory; only submission XLSX files go here")
    p.add_argument("--label-map", type=Path, help="Official XLSX or JSON; default is the supplied official 20-class mapping")
    p.add_argument("--aliases", type=Path, help="JSON of model-name alias -> official model name")
    p.add_argument("--details", type=Path, nargs="+", help="Task 2 slices.jsonl; otherwise automatically look beside each CSV")
    p.add_argument("--task2-level", choices=["burst", "track"], default="burst")
    p.add_argument("--unresolved", choices=["error", "singleton", "shared"], default="shared",
                   help="Default shared matches the requested common unknown ID for unresolved predictions")
    p.add_argument("--frequency-policy", choices=["error", "clip"], default="error")
    p.add_argument("--allow-partial-errors", action="store_true")
    p.add_argument("--input-root", type=Path, nargs="+", help="Optional local DAT roots for missing/extra file checks")
    p.add_argument("--check", action="store_true", help="Validate and write only the sidecar report, without XLSX files")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    report_path = output.parent/(output.name+"_report.json")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")
    files, report = build_plan(args)
    burst_complete = args.task2_level == "burst" or not any(ds.startswith("2") for ds, _ in files)
    report.update(output=str(output), report=str(report_path), valid_format=not report["errors"],
                  burst_complete=burst_complete, ready=not report["errors"] and burst_complete, written_workbooks=0)
    if not args.check and report["valid_format"]:
        output.mkdir(parents=True, exist_ok=True)
        task1 = defaultdict(list)
        for (dataset, sample), rows in sorted(files.items()):
            if dataset.startswith("1"):
                task1[dataset].append([sample, rows[0][0]])
            else:
                folder = output/dataset
                folder.mkdir(exist_ok=True)
                write_xlsx(folder/(sample+".xlsx"), ["无人机类别", "起始频率(Hz)", "终止频率(Hz)"], rows)
                report["written_workbooks"] += 1
        for dataset, rows in task1.items():
            write_xlsx(output/(dataset+".xlsx"), ["序号", "无人机类别"], rows)
            report["written_workbooks"] += 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid_format", "burst_complete", "ready", "datasets", "written_workbooks", "warnings", "report")}, ensure_ascii=False, indent=2))
    for message in report["errors"][:8]:
        print("ERROR: "+message)
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, InvalidOperation, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
