import cv2
from pathlib import Path

# =========================
# 配置区
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_SR = PROJECT_ROOT / "data" / "videos" / "sr.mp4"
VIDEO_IR = PROJECT_ROOT / "data" / "videos" / "ir.mp4"

OUT_SR = PROJECT_ROOT / "data" / "video_test" / "sr"
OUT_IR = PROJECT_ROOT / "data" / "video_test" / "ir"

# 每 5 秒抽 1 张
INTERVAL_SECONDS = 5


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_every_n_seconds(video_path: Path, out_dir: Path, interval_seconds: int):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    frame_interval = int(round(fps * interval_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0

    print(f"\nProcessing: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Extract one frame every {interval_seconds} seconds -> every {frame_interval} frames")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_name = f"{saved_idx:05d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames to: {out_dir}")


def main():
    ensure_dir(OUT_SR)
    ensure_dir(OUT_IR)

    extract_frames_every_n_seconds(VIDEO_SR, OUT_SR, INTERVAL_SECONDS)
    extract_frames_every_n_seconds(VIDEO_IR, OUT_IR, INTERVAL_SECONDS)

    print("\nDone.")
    print(f"SR frames: {OUT_SR}")
    print(f"IR frames: {OUT_IR}")


if __name__ == "__main__":
    main()