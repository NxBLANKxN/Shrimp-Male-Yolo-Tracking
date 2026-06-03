import argparse
import random
import re
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and organize them into a train/val dataset."
    )
    parser.add_argument(
        "--video-dir",
        default="video",
        help="Folder that contains source videos. Default: video",
    )
    parser.add_argument(
        "--output-dir",
        default="frame_dataset",
        help="Output dataset folder. Default: frame_dataset",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Save one frame every N frames. Default: 30",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of extracted images assigned to val. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split. Default: 42",
    )
    parser.add_argument(
        "--image-ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Saved image format. Default: jpg",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100. Used only when --image-ext jpg. Default: 95",
    )
    parser.add_argument(
        "--layout",
        choices=["images", "yolo", "classification"],
        default="images",
        help=(
            "Dataset layout. images: train/images, val/images. "
            "yolo: train/images, train/labels, val/images, val/labels. "
            "classification: train/male, train/female, val/male, val/female. Default: images"
        ),
    )
    parser.add_argument(
        "--empty-labels",
        action="store_true",
        help="With --layout yolo, also create an empty YOLO label .txt for every extracted image.",
    )
    return parser.parse_args()


def safe_stem(name):
    stem = Path(name).stem.strip()
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem or "video"


def collect_videos(video_dir):
    return sorted(
        path
        for path in Path(video_dir).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def infer_class_name(video_path):
    name = video_path.stem.lower()
    if any(token in name for token in ("母", "female", "f_")):
        return "female"
    if any(token in name for token in ("公", "male", "m_")):
        return "male"
    return "unknown"


def make_dataset_dirs(output_dir, layout):
    output_dir = Path(output_dir)
    if layout == "images":
        for split in ("train", "val"):
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    elif layout == "classification":
        for split in ("train", "val"):
            for class_name in ("male", "female", "unknown"):
                (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    elif layout == "yolo":
        for split in ("train", "val"):
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    return output_dir


def choose_split(rng, val_ratio):
    return "val" if rng.random() < val_ratio else "train"


def write_image(path, frame, image_ext, jpeg_quality):
    encode_ext = ".jpg" if image_ext == "jpg" else ".png"
    params = []
    if image_ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    ok, encoded = cv2.imencode(encode_ext, frame, params)
    if not ok:
        return False

    encoded.tofile(str(path))
    return path.exists() and path.stat().st_size > 0


def get_image_dir(output_dir, split, layout, class_name):
    if layout == "images":
        return output_dir / split / "images"
    if layout == "classification":
        return output_dir / split / class_name
    if layout == "yolo":
        return output_dir / split / "images"
    raise ValueError(f"Unsupported layout: {layout}")


def extract_video(
    video_path,
    output_dir,
    interval,
    val_ratio,
    rng,
    image_ext,
    jpeg_quality,
    layout,
    empty_labels,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return {"saved": 0, "failed": 0, "frames": 0}

    video_name = safe_stem(video_path.name)
    class_name = infer_class_name(video_path)
    frame_index = 0
    saved_index = 0
    failed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % interval == 0:
            split = choose_split(rng, val_ratio)
            image_name = f"{video_name}_{saved_index:05d}.{image_ext}"
            image_dir = get_image_dir(output_dir, split, layout, class_name)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / image_name

            if write_image(image_path, frame, image_ext, jpeg_quality):
                if layout == "yolo" and empty_labels:
                    label_path = output_dir / split / "labels" / f"{video_name}_{saved_index:05d}.txt"
                    label_path.write_text("", encoding="utf-8")
                saved_index += 1
            else:
                failed += 1
                print(f"[WARN] Failed to write image: {image_path}")

        frame_index += 1

    cap.release()
    return {"saved": saved_index, "failed": failed, "frames": frame_index}


def main():
    args = parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")
    if not 0 <= args.val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    videos = collect_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {video_dir}")

    output_dir = make_dataset_dirs(args.output_dir, args.layout)
    rng = random.Random(args.seed)

    total_saved = 0
    total_failed = 0

    print(f"Found {len(videos)} video(s) in {video_dir}")
    print(f"Output dataset: {output_dir}")
    print(f"Layout: {args.layout}")
    print(f"Saving every {args.interval} frame(s), val ratio: {args.val_ratio}")

    for video_path in videos:
        result = extract_video(
            video_path=video_path,
            output_dir=output_dir,
            interval=args.interval,
            val_ratio=args.val_ratio,
            rng=rng,
            image_ext=args.image_ext,
            jpeg_quality=args.jpeg_quality,
            layout=args.layout,
            empty_labels=args.empty_labels,
        )
        total_saved += result["saved"]
        total_failed += result["failed"]
        print(
            f"{video_path.name}: read {result['frames']} frame(s), "
            f"saved {result['saved']} image(s), failed {result['failed']}"
        )

    train_count = len(list((output_dir / "train").rglob(f"*.{args.image_ext}")))
    val_count = len(list((output_dir / "val").rglob(f"*.{args.image_ext}")))

    print("\nDone.")
    print(f"Total saved: {total_saved}, failed: {total_failed}")
    print(f"train/images: {train_count}")
    print(f"val/images: {val_count}")


if __name__ == "__main__":
    main()
