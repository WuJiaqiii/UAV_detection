"""Portable annotated PNGs and a local HTML report; no external web resources."""
import hashlib
from html import escape
from pathlib import Path

import numpy as np

from util.spectrogram_frontend import FrontendConfig, pixel_y_to_freq_mhz, render_rgb


COLORS = ["#ff6b6b", "#4dd4ac", "#64b5ff", "#ffc857", "#d597ff", "#ff9f68", "#76e5f2", "#cbe86b"]


def signal_color(observation):
    identity = observation.get("track_id", f"G{observation.get('group_id', 0)}")
    number = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    return COLORS[number % len(COLORS)]


def frequency_text(record, rf_center_mhz=None):
    absolute = rf_center_mhz is not None
    suffix = "absolute_mhz" if absolute else "offset_mhz"
    low, high = record["frequency_low_"+suffix], record["frequency_high_"+suffix]
    return f"{low:.3f} .. {high:.3f} MHz" + (" RF" if absolute else " offset")


def _font(size):
    from PIL import ImageFont
    for name in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_slice_preview(path, index, spectrum, domain, config, row, output_dir, rf_center_mhz=None):
    from PIL import Image, ImageDraw

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = render_rgb(spectrum, FrontendConfig.from_config(config), domain)
    height, width = rgb.shape[:2]
    left, top, panel = 100, 90, 410
    image = Image.new("RGB", (left+width+panel+30, top+height+65), (18, 23, 32))
    image.paste(Image.fromarray(rgb), (left, top))
    draw = ImageDraw.Draw(image)
    small, regular, title = _font(13), _font(16), _font(19)
    draw.text((15, 10), Path(path).name[:100], fill="white", font=title)
    draw.text((15, 39), f"Slice {index} | {row['slice_start_ms']:.3f} - {row['slice_end_ms']:.3f} ms | "
              f"{row['status']} | {len(row['groups'])} groups", fill="#cbd5e1", font=regular)
    axis_label = f"RF MHz (center {rf_center_mhz:g})" if rf_center_mhz is not None else "Offset MHz (RF center unknown)"
    draw.text((left, top-25), axis_label, fill="#cbd5e1", font=small)
    for y in np.linspace(0, height, 5):
        frequency = pixel_y_to_freq_mhz(y, height, config.sampling_rate, "fft") + (rf_center_mhz or 0.)
        draw.text((5, top+int(y)-8), f"{frequency:.3f}", fill="#cbd5e1", font=small)
    for x in np.linspace(0, width, 6):
        time = row["slice_start_ms"] + x/width * (row["slice_end_ms"]-row["slice_start_ms"])
        draw.text((left+int(x)-15, top+height+12), f"{time:.1f}", fill="#cbd5e1", font=small)
    draw.text((left+width-100, top+height+36), "File time (ms)", fill="#cbd5e1", font=small)
    sidebar = left+width+18
    for number, group in enumerate(row["groups"]):
        color = signal_color(group)
        for x1, y1, x2, y2 in group["selected_boxes"]:
            draw.rectangle((left+x1, top+y1, left+x2-1, top+y2-1), outline=color, width=2)
        first = group["selected_boxes"][0]
        draw.text((left+first[0]+2, top+max(0, first[1]-18)), group["track_id"], fill=color, font=small,
                  stroke_width=1, stroke_fill="black")
        if number < 7:
            y = top+number*69
            draw.text((sidebar, y), f"{group['track_id']} / G{group['group_id']}  {group['class_name'][:22]}", fill=color, font=regular)
            draw.text((sidebar, y+23), f"p={group['confidence']:.3f} | fc offset={group['frequency_center_offset_mhz']:.3f}", fill="#e2e8f0", font=small)
            draw.text((sidebar, y+43), frequency_text(group, rf_center_mhz), fill="#cbd5e1", font=small)
    if not row["groups"]:
        draw.text((sidebar, top), "No valid signal groups", fill="#cbd5e1", font=regular)
    if len(row["groups"]) > 7:
        draw.text((sidebar, top+height-22), "More groups: see signals.csv / report.html", fill="white", font=small)
    identity = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:12]
    destination = output_dir/f"{identity}_slice_{index:06d}.png"
    image.save(destination)
    return destination


def _timeline(result, observations, maximum_boxes=3000):
    pulses = [(observation, pulse) for observation in observations for pulse in observation["pulses"]]
    total = len(pulses)
    if total > maximum_boxes:
        pulses = [pulses[i] for i in np.linspace(0, total-1, maximum_boxes, dtype=int)]
    rf = result.get("rf_center_mhz")
    shift = rf or 0.
    low, high = shift-61.44, shift+61.44
    duration = ((result.get("sampling") or {}).get("total_samples", 0) /
                (result.get("sampling") or {}).get("sample_rate", 1) * 1000.)
    duration = max(duration, max((row["end_ms"] for row in observations), default=50.), 50.)
    width, height, left, top = 1000, 300, 100, 28
    x = lambda time: left+time/duration*width
    y = lambda frequency: top+(frequency-low)/(high-low)*height
    svg = [f'<svg viewBox="0 0 1140 385" role="img" aria-label="Signal frequency and file time">',
           f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#0b101a"/>']
    for frequency in np.linspace(low, high, 5):
        at = y(frequency)
        svg.append(f'<line x1="{left}" x2="{left+width}" y1="{at}" y2="{at}" stroke="#334155"/>')
        svg.append(f'<text x="4" y="{at+4}" fill="#cbd5e1" font-size="12">{frequency:.3f}</text>')
    for time in np.linspace(0., duration, 6):
        at = x(time)
        svg.append(f'<text x="{at-12}" y="350" fill="#cbd5e1" font-size="12">{time:.1f}</text>')
    for observation, pulse in pulses:
        a, b = pulse["frequency_low_offset_mhz"]+shift, pulse["frequency_high_offset_mhz"]+shift
        caption = (f"{observation['track_id']} | {observation['class_name']} p={observation['confidence']:.3f} | "
                   f"{pulse['start_ms']:.3f}-{pulse['end_ms']:.3f} ms | {a:.3f}-{b:.3f} MHz")
        svg.append(f'<rect x="{x(pulse["start_ms"]):.3f}" y="{y(a):.3f}" '
                   f'width="{max(.8, x(pulse["end_ms"])-x(pulse["start_ms"])):.3f}" '
                   f'height="{max(1., y(b)-y(a)):.3f}" fill="{signal_color(observation)}"><title>{escape(caption)}</title></rect>')
    svg.append(f'<text x="{left}" y="18" fill="#cbd5e1" font-size="12">'
               f'{"RF frequency (MHz)" if rf is not None else "Frequency offset (MHz); RF center unknown"}</text>')
    svg.append('<text x="480" y="380" fill="#cbd5e1" font-size="13">Time from file start (ms)</text></svg>')
    return "".join(svg)+f'<p class="muted">Timeline: {len(pulses)} / {total} detected boxes shown. All observations remain in signals.csv.</p>'


def write_html_report(output_dir, report, rows_by_file):
    output_dir = Path(output_dir)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Multi-signal evaluation</title>',
        '<style>body{font:15px system-ui,sans-serif;background:#111827;color:#e5e7eb;margin:32px auto;max-width:1280px;padding:0 18px}'
        'h1,h2{color:#fff}a{color:#7dd3fc}section{border-top:1px solid #475569;margin-top:32px;padding-top:12px}'
        'table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #334155;text-align:left;padding:9px}'
        '.scroll{overflow:auto}.muted{color:#94a3b8}img,svg{max-width:100%;height:auto}.previews{display:grid;gap:16px}'
        'details{margin:14px 0}code{overflow-wrap:anywhere}</style><body>',
        '<h1>Multi-signal evaluation</h1><p>Each group is classified independently. Track votes use time and frequency association; track IDs are candidate associations, not transmitter identities.</p>',
        '<p><a href="results.json">Full JSON</a> · <a href="signals.csv">Per-group CSV</a> · <a href="tracks.csv">Track votes CSV</a> · <a href="slices.jsonl">Slice records</a></p>',
        f'<p>{report["metrics"]["total_files"]} files · {report["metrics"]["total_observations"]} group observations · '
        f'{report["metrics"]["total_accepted_tracks"]} accepted tracks</p>']
    for result in report["files"]:
        rows = rows_by_file.get(result["path"], [])
        observations = [group for row in rows for group in row.get("groups", [])]
        parts += ['<section>', f'<h2>{escape(Path(result["path"]).name)}</h2>',
                  f'<p><code>{escape(result["path"])}</code></p>',
                  f'<p>Status: {escape(result["status"])} · Predicted classes: {escape(", ".join(result["predicted_classes"]) or "unknown")} · '
                  f'{result["total_slices"]} slices · {result["track_count"]} candidate tracks</p>']
        if result.get("error"):
            parts.append(f'<p>{escape(result["error"])}</p>')
        parts.append(_timeline(result, observations))
        parts.append('<h3>Track vote summaries</h3><div class="scroll"><table><tr><th>Track</th><th>Voted class</th><th>Time (ms)</th><th>Frequency range</th><th>Mean p / vote share</th><th>Valid / observed</th></tr>')
        for track in result["signals"]:
            mean, share = track["mean_probability"], track["vote_share"]
            probability = "unknown" if mean is None else f'{mean:.3f} / {share:.3f}'
            parts.append(f'<tr><td>{track["track_id"]}</td><td>{escape(track["class_name"] or "unknown")}</td>'
                         f'<td>{track["start_ms"]:.3f}–{track["end_ms"]:.3f}</td>'
                         f'<td>{escape(frequency_text(track, result.get("rf_center_mhz")))}</td>'
                         f'<td>{probability}</td><td>{track["valid_votes"]} / {track["observation_count"]}</td></tr>')
        parts.append('</table></div><details><summary>All per-group predictions and frequency positions</summary><div class="scroll"><table><tr><th>Slice / group</th><th>Track</th><th>Instant class</th><th>p</th><th>Time (ms)</th><th>Frequency range</th><th>Vote</th></tr>')
        for group in observations:
            parts.append(f'<tr><td>{group["slice_index"]} / {group["group_id"]}</td><td>{group["track_id"]}</td>'
                         f'<td>{escape(group["class_name"])}</td><td>{group["confidence"]:.3f}</td>'
                         f'<td>{group["start_ms"]:.3f}–{group["end_ms"]:.3f}</td>'
                         f'<td>{escape(frequency_text(group, result.get("rf_center_mhz")))}</td><td>{escape(group["vote_reason"])}</td></tr>')
        parts.append('</table></div></details><details open><summary>Annotated slices</summary><div class="previews">')
        for row in rows:
            if row.get("visualization"):
                relative = Path(row["visualization"]).as_posix()
                parts.append(f'<a href="{escape(relative, quote=True)}"><img loading="lazy" src="{escape(relative, quote=True)}" alt="Annotated slice {row["slice_index"]}"></a>')
        parts.append('</div></details></section>')
    parts.append('</body></html>')
    destination = output_dir/"report.html"
    destination.write_text("\n".join(parts), encoding="utf-8")
    return destination
