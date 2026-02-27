import os, re, csv, argparse, random
import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image

from util.preprocess import SignalPreprocessor
import util.detection as detlib

# 与 UAVDataset 同口径：protocol 在 "-[...]" 前
FNAME_RE = re.compile(r"""^(?P<protocol>.+?)-\[(?P<bracket>[^\]]+)\](?:-SNR-(?P<snr>[-+]?\d+(?:\.\d+)?))?""", re.VERBOSE)

def load_whitelist(csv_path: str) -> set[str]:
    keep = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Invalid CSV (no header): {csv_path}")
        col = "filename" if "filename" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            v = (row.get(col) or "").strip()
            if not v:
                continue
            v = os.path.basename(v).strip('"').strip("'")
            keep.add(v)
    return keep

def mat_in_whitelist(mat_fp: str, keep_names: set[str]) -> bool:
    fname = os.path.basename(mat_fp)                 # xxx.mat
    base = os.path.splitext(fname)[0]                # xxx
    png_name = base + ".png"                         # xxx.png
    return (fname in keep_names) or (png_name in keep_names) or (base in keep_names)

def get_protocol(mat_fp: str) -> str:
    m = FNAME_RE.search(os.path.basename(mat_fp))
    return m.group("protocol").strip() if m else "UNKNOWN"

def build_pt_index(cache_sig_dir: str) -> dict:
    """rel_path -> pt_path  (rel_path 是相对 dataset_path 的路径)"""
    rel2pt = {}
    for fn in os.listdir(cache_sig_dir):
        if not fn.endswith(".pt"):
            continue
        p = os.path.join(cache_sig_dir, fn)
        try:
            obj = torch.load(p, map_location="cpu")
            rel = (obj.get("source") or {}).get("rel", None)
            if rel:
                rel2pt[rel] = p
        except Exception:
            pass
    return rel2pt

def load_spec(mat_fp: str, mat_key: str) -> np.ndarray:
    mat = loadmat(mat_fp, variable_names=[mat_key])
    if mat_key not in mat:
        raise KeyError(f"key '{mat_key}' not in mat: {mat_fp}")
    return np.asarray(mat[mat_key], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--whitelist_csv", required=True)
    ap.add_argument("--cache_sig_dir", required=True, help="experiments/sam2_bbox_cache/<sig_hash>")
    ap.add_argument("--mat_key", default="summed_submatrices")

    ap.add_argument("--per_protocol_ratio", type=float, default=0.6)
    ap.add_argument("--min_per_protocol", type=int, default=2000)
    ap.add_argument("--max_per_protocol", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--boxes_from", choices=["filt", "raw"], default="filt")

    # 你要手动调的 post-filter 参数（不 sweep）
    ap.add_argument("--pre_min_area", type=int, default=20)
    ap.add_argument("--pre_min_ratio", type=float, default=1.5)
    ap.add_argument("--pre_freq_eps", type=float, default=5.0)
    ap.add_argument("--pre_freq_min_samples", type=int, default=2)
    ap.add_argument("--pre_nms_iou_thresh", type=float, default=0.5)

    ap.add_argument("--out_dir", default="experiments/postfilter_viz")
    args = ap.parse_args()

    random.seed(args.seed)

    keep = load_whitelist(args.whitelist_csv)

    mats = [os.path.join(args.dataset_path, f) for f in os.listdir(args.dataset_path) if f.endswith(".mat")]
    mats = [p for p in mats if mat_in_whitelist(p, keep)]

    # 按协议分组
    proto2mats = {}
    for p in mats:
        proto = get_protocol(p)
        proto2mats.setdefault(proto, []).append(p)

    # 抽样
    selected = []
    for proto, lst in sorted(proto2mats.items(), key=lambda x: x[0]):
        n = len(lst)
        k = int(round(n * args.per_protocol_ratio))
        k = max(args.min_per_protocol, min(args.max_per_protocol, k, n))
        random.shuffle(lst)
        selected.extend(lst[:k])
        print(f"[Select] {proto}: total={n}, keep={k}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 用 pt["source"]["rel"] 建索引（不依赖 pt 文件名）
    rel2pt = build_pt_index(args.cache_sig_dir)

    # 用你的参数构造 preprocessor（只用于 select_main_boxes）
    pre = SignalPreprocessor(
        sampling_rate=122.88e6, n_fft=512, hop_length=1,
        min_area=args.pre_min_area,
        min_ratio=args.pre_min_ratio,
        freq_eps=args.pre_freq_eps,
        freq_min_samples=args.pre_freq_min_samples,
        nms_iou_thresh=args.pre_nms_iou_thresh,
        use_patch_cnn=False,
    )

    # 统计：每个协议失败率/平均框数，方便你对比不同参数跑出来的效果
    stats = {}  # proto -> [k_list]
    fail = {}   # proto -> fail_count
    total = {}  # proto -> total_count

    for mat_fp in selected:
        rel = os.path.relpath(mat_fp, args.dataset_path)
        pt_path = rel2pt.get(rel, None)
        if pt_path is None:
            # 如果你缓存时 dataset_root 不是 dataset_path，这里可能对不上
            continue

        obj = torch.load(pt_path, map_location="cpu")
        raw = np.asarray(obj.get("raw_boxes", []), dtype=np.int32).reshape(-1, 4)
        filt = np.asarray(obj.get("filt_boxes", []), dtype=np.int32).reshape(-1, 4)
        det = filt if args.boxes_from == "filt" else raw

        clustered = pre.select_main_boxes(det)

        proto = get_protocol(mat_fp)
        total[proto] = total.get(proto, 0) + 1
        stats.setdefault(proto, []).append(int(clustered.shape[0]))
        if clustered.shape[0] == 0:
            fail[proto] = fail.get(proto, 0) + 1

        # 画图：raw / filtered / clustered
        spec = load_spec(mat_fp, args.mat_key)
        gray_u8 = detlib.to_uint8_grayscale(spec)
        rgb = np.stack([gray_u8] * 3, axis=-1)

        overlay = detlib.overlay_raw_filtered_clustered_boxes(
            rgb,
            raw_boxes_xyxy=raw,
            filt_boxes_xyxy=filt,
            cluster_boxes_xyxy=clustered,
        )

        out_sub = os.path.join(args.out_dir, proto)
        os.makedirs(out_sub, exist_ok=True)
        out_png = os.path.join(out_sub, os.path.splitext(os.path.basename(mat_fp))[0] + ".png")
        Image.fromarray(overlay).save(out_png)

    # 输出 summary.csv，方便你每次改参后做对比
    summ = os.path.join(args.out_dir, "summary.csv")
    with open(summ, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "n", "avg_k", "fail_rate(k=0)"])
        for proto in sorted(total.keys()):
            ks = stats.get(proto, [])
            n = total[proto]
            avg_k = float(np.mean(ks)) if ks else 0.0
            fr = float(fail.get(proto, 0) / max(1, n))
            w.writerow([proto, n, avg_k, fr])
    print("Saved overlays to:", args.out_dir)
    print("Saved summary to:", summ)

if __name__ == "__main__":
    main()