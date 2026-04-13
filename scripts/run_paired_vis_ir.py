import os
import sys
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINIMA_ROOT = PROJECT_ROOT / "minima"
ROMA_ROOT = MINIMA_ROOT / "third_party" / "RoMa_minima"

sys.path.insert(0, str(MINIMA_ROOT))
sys.path.insert(0, str(ROMA_ROOT))

from load_model import load_model  # noqa: E402
from src.utils.metrics import estimate_pose  # noqa: E402
from src.utils.plotting import make_matching_figure  # noqa: E402

METHOD = "loftr"   # 可选: sp_lg / loftr / roma

VIS_DIR = PROJECT_ROOT / "data" / "paired_test" / "original" / "vis"
IR_DIR = PROJECT_ROOT / "data" / "paired_test" / "original" / "ir"

BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paired_vis_ir"
RANSAC_THRESH = 3.0

# 不同方法对应的权重
CKPT_MAP = {
    "sp_lg": MINIMA_ROOT / "weights" / "minima_lightglue.pth",
    "loftr": MINIMA_ROOT / "weights" / "minima_loftr.ckpt",
    "roma": MINIMA_ROOT / "weights" / "minima_roma.pth",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_ir_name_from_vis(vis_name: str) -> str:
    """
    FLIR_00002_RGB.jpg -> FLIR_00002_PreviewData.jpeg
    """
    return vis_name.replace("_RGB.jpg", "_PreviewData.jpeg")


def dynamic_args(method: str):
    """
    构造 load_model 需要的参数对象。
    """
    ckpt = CKPT_MAP[method]
    args = SimpleNamespace()

    # 通用字段
    args.method = method
    args.ckpt = str(ckpt)
    
    if method == "loftr":
        args.thr = 0.2
    else:
        args.thr = None

    args.debug = False
    args.print_out = False
    args.save_figs = False
    args.svg = False
    args.exp_name = "paired_vis_ir"
    args.match_threshold = None
    args.max_keypoints = None
    args.width = None
    args.height = None
    args.device = "cuda"

    return args


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_simple_concat(path: Path, img0: np.ndarray, img1: np.ndarray) -> None:
    """
    保存简单拼接图，便于快速检查输入。
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    target_h = min(h0, h1)
    if h0 != target_h:
        img0 = cv2.resize(img0, (int(w0 * target_h / h0), target_h))
    if h1 != target_h:
        img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))

    concat = np.concatenate([img0, img1], axis=1)
    concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), concat_bgr)


def save_match_figure(
    path: Path,
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    text_lines,
) -> None:
    """
    用 minima 自带 plotting 画线。
    """
    n = len(mkpts0)
    if n == 0:
        save_simple_concat(path, img0, img1)
        return

    # 统一颜色：绿色
    color = np.tile(np.array([[0.0, 1.0, 0.0, 1.0]]), (n, 1))

    make_matching_figure(
        img0,
        img1,
        mkpts0,
        mkpts1,
        color,
        text=text_lines,
        path=str(path),
        dpi=100,
        svg=False
    )


def safe_inlier_mask(inliers, n_matches: int) -> np.ndarray:
    """
    把 estimate_pose 返回的 inliers 统一成 bool mask。
    """
    if inliers is None:
        return np.zeros((n_matches,), dtype=bool)

    inliers = np.asarray(inliers)

    if inliers.dtype == np.bool_:
        if inliers.ndim == 1 and len(inliers) == n_matches:
            return inliers
        return inliers.reshape(-1).astype(bool)

    # 有些实现返回 0/1 mask
    inliers = inliers.reshape(-1)
    if len(inliers) == n_matches:
        return inliers.astype(bool)

    # 保底
    mask = np.zeros((n_matches,), dtype=bool)
    usable = min(len(inliers), n_matches)
    mask[:usable] = inliers[:usable].astype(bool)
    return mask


def run_one_pair(matcher, vis_path: Path, ir_path: Path):
    """
    对单对图像跑匹配 + RANSAC。
    """
    # 这套数据没有 K / dist / GT，所以先传 None
    match_res = matcher(str(vis_path), str(ir_path), None, None, None, None)

    mkpts0 = np.asarray(match_res["mkpts0"])
    mkpts1 = np.asarray(match_res["mkpts1"])
    num_matches = len(mkpts0)

    result = {
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "matches": num_matches,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "H": None,
        "inlier_mask": np.zeros((num_matches,), dtype=bool),
    }

    if num_matches < 4:
        return result

    # estimate_pose 内部本来更偏相对位姿估计，通常需要 K；
    # 你的数据没有 K，所以这里优先尝试 OpenCV 的单应性估计。
    H, inlier_mask_cv = cv2.findHomography(
        mkpts0,
        mkpts1,
        cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH
    )

    if H is None or inlier_mask_cv is None:
        return result

    inlier_mask = safe_inlier_mask(inlier_mask_cv, num_matches)
    num_inliers = int(inlier_mask.sum())
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0

    result.update({
        "inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "H": H,
        "inlier_mask": inlier_mask,
    })
    return result


def main():
    if METHOD not in CKPT_MAP:
        raise ValueError(f"Unsupported METHOD: {METHOD}")

    if not VIS_DIR.exists():
        raise FileNotFoundError(f"VIS_DIR not found: {VIS_DIR}")
    if not IR_DIR.exists():
        raise FileNotFoundError(f"IR_DIR not found: {IR_DIR}")

    method_output_dir = BASE_OUTPUT_DIR / METHOD
    ensure_dir(method_output_dir)

    args = dynamic_args(METHOD)
    matcher = load_model(METHOD, args)

    vis_files = sorted([
        f for f in os.listdir(VIS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    rows = []

    for vis_name in tqdm(vis_files, desc=f"Running {METHOD}"):
        vis_path = VIS_DIR / vis_name
        ir_name = build_ir_name_from_vis(vis_name)
        ir_path = IR_DIR / ir_name

        if not ir_path.exists():
            print(f"[WARN] Missing IR pair for {vis_name}: {ir_name}")
            continue

        try:
            result = run_one_pair(matcher, vis_path, ir_path)

            num_matches = result["matches"]
            num_inliers = result["inliers"]
            inlier_ratio = result["inlier_ratio"]
            inlier_mask = result["inlier_mask"]
            H = result["H"]

            print(
                f"{vis_name}: matches = {num_matches}, "
                f"inliers = {num_inliers}, "
                f"inlier_ratio = {inlier_ratio:.3f}"
            )

            img0 = read_rgb(vis_path)
            img1 = read_rgb(ir_path)

            stem = vis_name.rsplit(".", 1)[0]

            # 0) 原始拼接图
            concat_path = method_output_dir / f"{stem}_concat.jpg"
            save_simple_concat(concat_path, img0, img1)

            # 1) before_ransac
            before_path = method_output_dir / f"{stem}_before_ransac.jpg"
            before_text = [
                f"method: {METHOD}",
                f"matches: {num_matches}",
            ]
            save_match_figure(
                before_path,
                img0,
                img1,
                result["mkpts0"],
                result["mkpts1"],
                before_text
            )

            # 2) after_ransac
            after_path = method_output_dir / f"{stem}_after_ransac.jpg"
            if num_inliers > 0:
                after_text = [
                    f"method: {METHOD}",
                    f"matches: {num_matches}",
                    f"inliers: {num_inliers}",
                    f"inlier_ratio: {inlier_ratio:.3f}",
                ]
                save_match_figure(
                    after_path,
                    img0,
                    img1,
                    result["mkpts0"][inlier_mask],
                    result["mkpts1"][inlier_mask],
                    after_text
                )
            else:
                save_simple_concat(after_path, img0, img1)

            # 3) 保存 warp 结果（把 vis 变到 ir）
            if H is not None:
                h1, w1 = img1.shape[:2]
                warped = cv2.warpPerspective(
                    cv2.cvtColor(img0, cv2.COLOR_RGB2BGR),
                    H,
                    (w1, h1)
                )
                warp_path = method_output_dir / f"{stem}_warp_to_ir.jpg"
                cv2.imwrite(str(warp_path), warped)

            row = {
                "vis_name": vis_name,
                "ir_name": ir_name,
                "matches": num_matches,
                "inliers": num_inliers,
                "inlier_ratio": f"{inlier_ratio:.6f}",
                "status": "ok",
            }

        except Exception as e:
            print(f"[ERROR] {vis_name}: {e}")
            row = {
                "vis_name": vis_name,
                "ir_name": ir_name,
                "matches": -1,
                "inliers": -1,
                "inlier_ratio": "",
                "status": f"error: {e}",
            }

        rows.append(row)

    # summary.csv
    summary_csv = method_output_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vis_name",
                "ir_name",
                "matches",
                "inliers",
                "inlier_ratio",
                "status",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if isinstance(r["matches"], int) and r["matches"] >= 0]

    if valid_rows:
        avg_matches = sum(r["matches"] for r in valid_rows) / len(valid_rows)
        avg_inliers = sum(r["inliers"] for r in valid_rows) / len(valid_rows)
        avg_inlier_ratio = sum(float(r["inlier_ratio"]) for r in valid_rows) / len(valid_rows)
    else:
        avg_matches = 0.0
        avg_inliers = 0.0
        avg_inlier_ratio = 0.0

    print("\n===== RESULT =====")
    print(f"Method: {METHOD}")
    print(f"Pairs processed: {len(valid_rows)}")
    print(f"Average matches: {avg_matches:.2f}")
    print(f"Average inliers: {avg_inliers:.2f}")
    print(f"Average inlier ratio: {avg_inlier_ratio:.4f}")
    print(f"Saved to: {method_output_dir}")


if __name__ == "__main__":
    main()