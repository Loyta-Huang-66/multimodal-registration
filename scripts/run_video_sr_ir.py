import os
import sys
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm

# =========================
# 路径设置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINIMA_ROOT = PROJECT_ROOT / "minima"

sys.path.insert(0, str(MINIMA_ROOT))

from load_model import load_model  # noqa: E402
from src.utils.plotting import make_matching_figure  # noqa: E402


# =========================
# 配置区
# =========================
METHOD = "sp_lg"   # 可选: sp_lg / loftr

SR_DIR = PROJECT_ROOT / "data" / "video_test" / "sr"
IR_DIR = PROJECT_ROOT / "data" / "video_test" / "ir"

BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "video_srir"

RANSAC_THRESH = 3.0

CKPT_MAP = {
    "sp_lg": MINIMA_ROOT / "weights" / "minima_lightglue.pth",
    "loftr": MINIMA_ROOT / "weights" / "minima_loftr.ckpt",
}


# =========================
# 工具函数
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_ir_name_from_sr(sr_name: str) -> str:
    """
    sr_00000.jpg -> ir_00000.jpg
    """
    return sr_name.replace("sr_", "ir_")


def dynamic_args(method: str):
    args = SimpleNamespace()
    args.method = method
    args.ckpt = str(CKPT_MAP[method])

    if method == "loftr":
        args.thr = 0.2
    else:
        args.thr = None

    args.debug = False
    args.print_out = False
    args.save_figs = False
    args.svg = False
    args.exp_name = "video_srir"
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
    n = len(mkpts0)
    if n == 0:
        save_simple_concat(path, img0, img1)
        return

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


def safe_inlier_mask(mask, n_matches: int) -> np.ndarray:
    if mask is None:
        return np.zeros((n_matches,), dtype=bool)

    mask = np.asarray(mask).reshape(-1)
    if len(mask) != n_matches:
        usable = min(len(mask), n_matches)
        out = np.zeros((n_matches,), dtype=bool)
        out[:usable] = mask[:usable].astype(bool)
        return out

    return mask.astype(bool)


def run_one_pair(matcher, sr_path: Path, ir_path: Path):
    match_res = matcher(str(sr_path), str(ir_path), None, None, None, None)

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


# =========================
# 主流程
# =========================
def main():
    if METHOD not in CKPT_MAP:
        raise ValueError(f"Unsupported METHOD: {METHOD}")

    method_output_dir = BASE_OUTPUT_DIR / METHOD
    ensure_dir(method_output_dir)

    args = dynamic_args(METHOD)
    matcher = load_model(METHOD, args)

    # 读取 sr 目录下全部帧
    sr_files = sorted([
        f for f in os.listdir(SR_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    rows = []

    for sr_name in tqdm(sr_files, desc=f"Running {METHOD} on all video SR-IR pairs"):
        sr_path = SR_DIR / sr_name
        ir_name = build_ir_name_from_sr(sr_name)
        ir_path = IR_DIR / ir_name

        if not sr_path.exists():
            print(f"[WARN] Missing SR frame: {sr_path}")
            continue
        if not ir_path.exists():
            print(f"[WARN] Missing IR frame: {ir_path}")
            continue

        try:
            result = run_one_pair(matcher, sr_path, ir_path)

            num_matches = result["matches"]
            num_inliers = result["inliers"]
            inlier_ratio = result["inlier_ratio"]
            inlier_mask = result["inlier_mask"]
            H = result["H"]

            print(
                f"{sr_name}: matches = {num_matches}, "
                f"inliers = {num_inliers}, "
                f"inlier_ratio = {inlier_ratio:.3f}"
            )

            img0 = read_rgb(sr_path)
            img1 = read_rgb(ir_path)

            stem = sr_name.rsplit(".", 1)[0]

            concat_path = method_output_dir / f"{stem}_concat.jpg"
            before_path = method_output_dir / f"{stem}_before_ransac.jpg"
            after_path = method_output_dir / f"{stem}_after_ransac.jpg"
            warp_path = method_output_dir / f"{stem}_warp_to_ir.jpg"

            save_simple_concat(concat_path, img0, img1)

            save_match_figure(
                before_path,
                img0,
                img1,
                result["mkpts0"],
                result["mkpts1"],
                [
                    f"method: {METHOD}",
                    f"matches: {num_matches}",
                ]
            )

            if num_inliers > 0:
                save_match_figure(
                    after_path,
                    img0,
                    img1,
                    result["mkpts0"][inlier_mask],
                    result["mkpts1"][inlier_mask],
                    [
                        f"method: {METHOD}",
                        f"matches: {num_matches}",
                        f"inliers: {num_inliers}",
                        f"inlier_ratio: {inlier_ratio:.3f}",
                    ]
                )
            else:
                save_simple_concat(after_path, img0, img1)

            if H is not None:
                h1, w1 = img1.shape[:2]

                # 1. 将短波红外图像映射到长波红外坐标系
                warped = cv2.warpPerspective(
                    cv2.cvtColor(img0, cv2.COLOR_RGB2BGR),
                    H,
                    (w1, h1)
                )

                # 保存单独变换结果
                cv2.imwrite(str(warp_path), warped)

                # 2. 读取长波红外参考图像
                ir_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

                # 3. 融合图像
                fusion = cv2.addWeighted(
                    warped, 0.5,
                    ir_bgr, 0.5,
                    0
                )

                # 4. 保存融合结果
                fusion_path = method_output_dir / f"{stem}_fusion_to_ir.jpg"
                cv2.imwrite(str(fusion_path), fusion)

            rows.append({
                "sr_name": sr_name,
                "ir_name": ir_name,
                "matches": num_matches,
                "inliers": num_inliers,
                "inlier_ratio": f"{inlier_ratio:.6f}",
                "status": "ok",
            })

        except Exception as e:
            print(f"[ERROR] {sr_name}: {e}")
            rows.append({
                "sr_name": sr_name,
                "ir_name": ir_name,
                "matches": -1,
                "inliers": -1,
                "inlier_ratio": "",
                "status": f"error: {e}",
            })

    summary_csv = method_output_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sr_name", "ir_name", "matches", "inliers", "inlier_ratio", "status"]
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

    print("\n===== FINAL RESULT =====")
    print(f"Method: {METHOD}")
    print(f"Pairs processed: {len(valid_rows)}")
    print(f"Average matches: {avg_matches:.2f}")
    print(f"Average inliers: {avg_inliers:.2f}")
    print(f"Average inlier ratio: {avg_inlier_ratio:.4f}")
    print(f"Saved to: {method_output_dir}")


if __name__ == "__main__":
    main()