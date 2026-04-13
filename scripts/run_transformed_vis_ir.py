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
METHOD = "loftr"   # 可选: sp_lg / loftr

VIS_DIR = PROJECT_ROOT / "data" / "paired_test" / "original" / "vis"
IR_DIR = PROJECT_ROOT / "data" / "paired_test" / "original" / "ir"

BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transformed_vis_ir"

RANSAC_THRESH = 3.0

# 先做单一变换实验
TRANSFORMS = [
    {"name": "translate_x20_y10", "type": "translate", "tx": 20, "ty": 10},
    {"name": "rotate_5deg", "type": "rotate", "angle": 5},
    {"name": "rotate_10deg", "type": "rotate", "angle": 10},
    {"name": "scale_1.1", "type": "scale", "scale": 1.1},
]

CKPT_MAP = {
    "sp_lg": MINIMA_ROOT / "weights" / "minima_lightglue.pth",
    "loftr": MINIMA_ROOT / "weights" / "minima_loftr.ckpt",
}


# =========================
# 基础函数
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_ir_name_from_vis(vis_name: str) -> str:
    return vis_name.replace("_RGB.jpg", "_PreviewData.jpeg")


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
    args.exp_name = "transformed_vis_ir"
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def save_simple_concat(path: Path, img0: np.ndarray, img1: np.ndarray) -> None:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    target_h = min(h0, h1)

    if h0 != target_h:
        img0 = cv2.resize(img0, (int(w0 * target_h / h0), target_h))
    if h1 != target_h:
        img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))

    concat = np.concatenate([img0, img1], axis=1)
    save_rgb(path, concat)


def save_match_figure(path: Path, img0: np.ndarray, img1: np.ndarray,
                      mkpts0: np.ndarray, mkpts1: np.ndarray, text_lines):
    n = len(mkpts0)
    if n == 0:
        save_simple_concat(path, img0, img1)
        return

    color = np.tile(np.array([[0.0, 1.0, 0.0, 1.0]]), (n, 1))
    make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        text=text_lines, path=str(path), dpi=100, svg=False
    )


def apply_transform_to_vis(img: np.ndarray, transform_cfg: dict):
    """
    返回:
    - transformed_img
    - H_gt (3x3)
    """
    h, w = img.shape[:2]

    if transform_cfg["type"] == "translate":
        tx = transform_cfg["tx"]
        ty = transform_cfg["ty"]
        H = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)

    elif transform_cfg["type"] == "rotate":
        angle = transform_cfg["angle"]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 2x3
        H = np.vstack([M, [0, 0, 1]]).astype(np.float32)

    elif transform_cfg["type"] == "scale":
        s = transform_cfg["scale"]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 0, s)
        H = np.vstack([M, [0, 0, 1]]).astype(np.float32)

    else:
        raise ValueError(f"Unknown transform type: {transform_cfg['type']}")

    transformed = cv2.warpPerspective(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        H,
        (w, h)
    )
    transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    return transformed, H


def compute_reprojection_error(H_gt: np.ndarray, H_est: np.ndarray, w: int, h: int):
    """
    用图像四角点比较 H_gt 和 H_est 的重投影误差
    """
    pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    gt_pts = cv2.perspectiveTransform(pts, H_gt)
    est_pts = cv2.perspectiveTransform(pts, H_est)

    err = np.linalg.norm(gt_pts - est_pts, axis=2).mean()
    return float(err)


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


def main():
    if METHOD not in CKPT_MAP:
        raise ValueError(f"Unsupported METHOD: {METHOD}")

    args = dynamic_args(METHOD)
    matcher = load_model(METHOD, args)

    vis_files = sorted([
        f for f in os.listdir(VIS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:3]

    method_output_dir = BASE_OUTPUT_DIR / METHOD
    ensure_dir(method_output_dir)

    rows = []

    for transform_cfg in TRANSFORMS:
        transform_name = transform_cfg["name"]
        transform_dir = method_output_dir / transform_name
        ensure_dir(transform_dir)

        for vis_name in tqdm(vis_files, desc=f"{METHOD} | {transform_name}"):
            vis_path = VIS_DIR / vis_name
            ir_name = build_ir_name_from_vis(vis_name)
            ir_path = IR_DIR / ir_name

            if not ir_path.exists():
                print(f"[WARN] Missing IR pair for {vis_name}")
                continue

            try:
                img_vis = read_rgb(vis_path)
                img_ir = read_rgb(ir_path)

                # 1. 对 VIS 施加已知几何变换
                img_vis_tf, H_gt = apply_transform_to_vis(img_vis, transform_cfg)

                stem = vis_name.rsplit(".", 1)[0]

                transformed_vis_path = transform_dir / f"{stem}_transformed_vis.jpg"
                save_rgb(transformed_vis_path, img_vis_tf)

                # 2. 跑匹配：变换后的 VIS vs 原始 IR
                match_res = matcher(
                    str(transformed_vis_path),
                    str(ir_path),
                    None, None, None, None
                )

                mkpts0 = np.asarray(match_res["mkpts0"])
                mkpts1 = np.asarray(match_res["mkpts1"])
                num_matches = len(mkpts0)

                if num_matches < 4:
                    rows.append({
                        "image": vis_name,
                        "method": METHOD,
                        "transform": transform_name,
                        "matches": num_matches,
                        "inliers": 0,
                        "inlier_ratio": 0.0,
                        "reprojection_error": "",
                        "status": "too_few_matches"
                    })
                    continue

                # 3. 用 RANSAC 估计单应矩阵
                H_est, inlier_mask_cv = cv2.findHomography(
                    mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH
                )

                if H_est is None or inlier_mask_cv is None:
                    rows.append({
                        "image": vis_name,
                        "method": METHOD,
                        "transform": transform_name,
                        "matches": num_matches,
                        "inliers": 0,
                        "inlier_ratio": 0.0,
                        "reprojection_error": "",
                        "status": "homography_failed"
                    })
                    continue

                inlier_mask = safe_inlier_mask(inlier_mask_cv, num_matches)
                num_inliers = int(inlier_mask.sum())
                inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0

                # 4. 计算重投影误差
                h, w = img_vis.shape[:2]
                reproj_err = compute_reprojection_error(H_gt, H_est, w, h)

                # 5. 保存 before/after/warp
                before_path = transform_dir / f"{stem}_before_ransac.jpg"
                after_path = transform_dir / f"{stem}_after_ransac.jpg"
                warp_path = transform_dir / f"{stem}_warp_to_ir.jpg"

                save_match_figure(
                    before_path,
                    img_vis_tf,
                    img_ir,
                    mkpts0,
                    mkpts1,
                    [
                        f"method: {METHOD}",
                        f"transform: {transform_name}",
                        f"matches: {num_matches}",
                    ]
                )

                save_match_figure(
                    after_path,
                    img_vis_tf,
                    img_ir,
                    mkpts0[inlier_mask],
                    mkpts1[inlier_mask],
                    [
                        f"method: {METHOD}",
                        f"transform: {transform_name}",
                        f"matches: {num_matches}",
                        f"inliers: {num_inliers}",
                        f"inlier_ratio: {inlier_ratio:.3f}",
                        f"reproj_err: {reproj_err:.3f}",
                    ]
                )

                h_ir, w_ir = img_ir.shape[:2]
                warped = cv2.warpPerspective(
                    cv2.cvtColor(img_vis_tf, cv2.COLOR_RGB2BGR),
                    H_est,
                    (w_ir, h_ir)
                )
                cv2.imwrite(str(warp_path), warped)

                rows.append({
                    "image": vis_name,
                    "method": METHOD,
                    "transform": transform_name,
                    "matches": num_matches,
                    "inliers": num_inliers,
                    "inlier_ratio": f"{inlier_ratio:.6f}",
                    "reprojection_error": f"{reproj_err:.6f}",
                    "status": "ok"
                })

                print(
                    f"{vis_name} | {transform_name}: "
                    f"matches={num_matches}, inliers={num_inliers}, "
                    f"inlier_ratio={inlier_ratio:.3f}, reproj_err={reproj_err:.3f}"
                )

            except Exception as e:
                print(f"[ERROR] {vis_name} | {transform_name}: {e}")
                rows.append({
                    "image": vis_name,
                    "method": METHOD,
                    "transform": transform_name,
                    "matches": -1,
                    "inliers": -1,
                    "inlier_ratio": "",
                    "reprojection_error": "",
                    "status": f"error: {e}"
                })

    # 6. 保存 summary.csv
    summary_csv = method_output_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "method",
                "transform",
                "matches",
                "inliers",
                "inlier_ratio",
                "reprojection_error",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        r for r in rows
        if isinstance(r["matches"], int) and r["matches"] >= 0 and r["status"] == "ok"
    ]

    if valid_rows:
        avg_matches = sum(r["matches"] for r in valid_rows) / len(valid_rows)
        avg_inliers = sum(r["inliers"] for r in valid_rows) / len(valid_rows)
        avg_inlier_ratio = sum(float(r["inlier_ratio"]) for r in valid_rows) / len(valid_rows)
        avg_reproj_err = sum(float(r["reprojection_error"]) for r in valid_rows) / len(valid_rows)
    else:
        avg_matches = 0.0
        avg_inliers = 0.0
        avg_inlier_ratio = 0.0
        avg_reproj_err = 0.0

    print("\n===== FINAL RESULT =====")
    print(f"Method: {METHOD}")
    print(f"Valid samples: {len(valid_rows)}")
    print(f"Average matches: {avg_matches:.2f}")
    print(f"Average inliers: {avg_inliers:.2f}")
    print(f"Average inlier ratio: {avg_inlier_ratio:.4f}")
    print(f"Average reprojection error: {avg_reproj_err:.4f}")
    print(f"Saved to: {method_output_dir}")


if __name__ == "__main__":
    main()