import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
HBB_CLASS_NAMES = ["male_line"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use a trained OBB model to crop single shrimp images for later HBB annotation/training."
    )
    parser.add_argument(
        "--model",
        default="runs/obb/traindata20260602_公母蝦_obb_1/weights/best.pt",
        help="Trained OBB model path. Default: runs/obb/traindata20260602_公母蝦_obb_1/weights/best.pt",
    )
    parser.add_argument(
        "--video-dir",
        default="../video",
        help="Folder that contains source videos. Default: ../video",
    )
    parser.add_argument(
        "--output-dir",
        default="../HBB/shrimp_HBB_dataset_from_obb",
        help="Output HBB dataset folder. Default: ../HBB/shrimp_HBB_dataset_from_obb",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=180,
        help="Run OBB inference once every N frames. Default: 180",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of cropped shrimp images assigned to val. Default: 0.2",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="OBB inference image size. Use the same or larger size as OBB training. Default: 1280",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum OBB detection confidence. Default: 0.35",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.12,
        help="Extra crop margin around each OBB box, as a ratio of box size. Default: 0.12",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=40,
        help="Skip crops whose width or height is smaller than this many pixels. Default: 40",
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
        "--enhance",
        action="store_true",
        help="Apply CLAHE + sharpening to cropped images.",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "keep"],
        default="horizontal",
        help="Crop orientation. horizontal rotates tall crops to landscape. keep preserves OBB output. Default: horizontal",
    )
    parser.add_argument(
        "--empty-labels",
        action="store_true",
        help="Create an empty YOLO HBB .txt label file for each cropped image.",
    )
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Also write HBB/data.yaml pointing to this cropped dataset.",
    )
    return parser.parse_args()


def safe_stem(name):
    stem = Path(name).stem.strip()
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem or "video"


def resolve_path(path, base_dir):
    path = Path(path)
    return path if path.is_absolute() else base_dir / path


def collect_videos(video_dir):
    video_dir = Path(video_dir)
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def make_dataset_dirs(output_dir):
    output_dir = Path(output_dir)
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


def crop_oriented_box(image, obb_box, padding):
    cx, cy, w, h, rotation = obb_box
    crop_w = max(1, int(round(w * (1 + padding * 2))))
    crop_h = max(1, int(round(h * (1 + padding * 2))))
    angle = np.degrees(rotation)

    rect = ((float(cx), float(cy)), (float(crop_w), float(crop_h)), float(angle))
    src = cv2.boxPoints(rect).astype("float32")
    dst = np.array(
        [
            [0, crop_h - 1],
            [0, 0],
            [crop_w - 1, 0],
            [crop_w - 1, crop_h - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (crop_w, crop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize_crop_orientation(image, orientation):
    if orientation == "keep":
        return image

    height, width = image.shape[:2]
    if height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def sharpen_and_contrast(image):
    if image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1.2)
    return cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0)


def iter_obb_detections(result, conf_threshold):
    if result.obb is None:
        return

    xywhr = result.obb.xywhr.cpu().numpy()
    confs = result.obb.conf.cpu().numpy() if result.obb.conf is not None else np.ones(len(xywhr))
    classes = result.obb.cls.cpu().numpy() if result.obb.cls is not None else np.zeros(len(xywhr))

    for box, conf, cls_id in zip(xywhr, confs, classes):
        if float(conf) >= conf_threshold:
            yield box, float(conf), int(cls_id)


def extract_crops(
    model,
    video_path,
    output_dir,
    interval,
    val_ratio,
    rng,
    imgsz,
    conf,
    padding,
    min_size,
    image_ext,
    jpeg_quality,
    enhance,
    orientation,
    empty_labels,
    metadata_writer,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return {"frames": 0, "saved": 0, "failed": 0, "detections": 0}

    video_name = safe_stem(video_path.name)
    frame_index = 0
    crop_index = 0
    saved = 0
    failed = 0
    detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % interval == 0:
            results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
            for det_index, (obb_box, det_conf, cls_id) in enumerate(iter_obb_detections(results[0], conf)):
                detections += 1

                try:
                    crop = crop_oriented_box(frame, obb_box, padding)
                except cv2.error:
                    failed += 1
                    continue

                crop = normalize_crop_orientation(crop, orientation)
                height, width = crop.shape[:2]
                if width < min_size or height < min_size:
                    continue

                if enhance:
                    crop = sharpen_and_contrast(crop)

                split = choose_split(rng, val_ratio)
                image_stem = f"{video_name}_crop_{crop_index:05d}"
                image_path = output_dir / split / "images" / f"{image_stem}.{image_ext}"

                if write_image(image_path, crop, image_ext, jpeg_quality):
                    if empty_labels:
                        label_path = output_dir / split / "labels" / f"{image_stem}.txt"
                        label_path.write_text("", encoding="utf-8")

                    cx, cy, obb_w, obb_h, rotation = obb_box
                    metadata_writer.writerow(
                        {
                            "image": str(image_path),
                            "split": split,
                            "source_video": str(video_path),
                            "source_frame": frame_index,
                            "detection_index": det_index,
                            "obb_class": cls_id,
                            "obb_conf": f"{det_conf:.6f}",
                            "obb_cx": f"{cx:.3f}",
                            "obb_cy": f"{cy:.3f}",
                            "obb_w": f"{obb_w:.3f}",
                            "obb_h": f"{obb_h:.3f}",
                            "obb_rotation": f"{rotation:.6f}",
                            "crop_width": width,
                            "crop_height": height,
                        }
                    )
                    saved += 1
                    crop_index += 1
                else:
                    failed += 1
                    print(f"[WARN] Failed to write image: {image_path}")

        frame_index += 1

    cap.release()
    return {"frames": frame_index, "saved": saved, "failed": failed, "detections": detections}


def write_hbb_yaml(hbb_dir, output_dir):
    yaml_path = hbb_dir / "data.yaml"
    try:
        dataset_path = output_dir.relative_to(hbb_dir).as_posix()
    except ValueError:
        dataset_path = output_dir.as_posix()

    names = ", ".join(f"'{name}'" for name in HBB_CLASS_NAMES)
    yaml_text = (
        f"train: ./{dataset_path}/train/images\n"
        f"val: ./{dataset_path}/val/images\n\n"
        f"nc: {len(HBB_CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    hbb_dir = script_dir.parent / "HBB"

    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")
    if not 0 <= args.val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")
    if args.padding < 0:
        raise ValueError("--padding must be greater than or equal to 0")
    if args.min_size <= 0:
        raise ValueError("--min-size must be greater than 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    model_path = resolve_path(args.model, script_dir)
    video_dir = resolve_path(args.video_dir, script_dir)
    output_dir = make_dataset_dirs(resolve_path(args.output_dir, script_dir))

    if not model_path.exists():
        raise FileNotFoundError(f"OBB model not found: {model_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    videos = collect_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in: {video_dir}")

    print(f"Loading OBB model: {model_path}")
    model = YOLO(str(model_path))
    rng = random.Random(args.seed)

    metadata_path = output_dir / "crop_metadata.csv"
    fieldnames = [
        "image",
        "split",
        "source_video",
        "source_frame",
        "detection_index",
        "obb_class",
        "obb_conf",
        "obb_cx",
        "obb_cy",
        "obb_w",
        "obb_h",
        "obb_rotation",
        "crop_width",
        "crop_height",
    ]

    total_saved = 0
    total_failed = 0
    total_detections = 0

    print(f"Found {len(videos)} video(s) in {video_dir}")
    print(f"Output HBB dataset: {output_dir}")
    print(f"Inference every {args.interval} frame(s), conf: {args.conf}, val ratio: {args.val_ratio}")
    print("For HBB training, label each cropped image with YOLO HBB boxes:")
    print("class x_center y_center width height")
    print("Coordinates must be normalized to 0-1.\n")

    with metadata_path.open("w", newline="", encoding="utf-8-sig") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        writer.writeheader()

        for video_path in videos:
            result = extract_crops(
                model=model,
                video_path=video_path,
                output_dir=output_dir,
                interval=args.interval,
                val_ratio=args.val_ratio,
                rng=rng,
                imgsz=args.imgsz,
                conf=args.conf,
                padding=args.padding,
                min_size=args.min_size,
                image_ext=args.image_ext,
                jpeg_quality=args.jpeg_quality,
                enhance=args.enhance,
                orientation=args.orientation,
                empty_labels=args.empty_labels,
                metadata_writer=writer,
            )
            total_saved += result["saved"]
            total_failed += result["failed"]
            total_detections += result["detections"]
            print(
                f"{video_path.name}: read {result['frames']} frame(s), "
                f"detections {result['detections']}, saved {result['saved']} crop(s), "
                f"failed {result['failed']}"
            )

    if args.write_yaml:
        yaml_path = write_hbb_yaml(hbb_dir, output_dir)
        print(f"\nWrote HBB data config: {yaml_path}")

    train_count = len(list((output_dir / "train" / "images").glob(f"*.{args.image_ext}")))
    val_count = len(list((output_dir / "val" / "images").glob(f"*.{args.image_ext}")))

    print("\nDone.")
    print(f"Total detections: {total_detections}")
    print(f"Total saved crops: {total_saved}, failed: {total_failed}")
    print(f"train/images: {train_count}")
    print(f"val/images: {val_count}")
    print(f"Metadata: {metadata_path}")
    print("\nNext step: annotate HBB labels in train/labels and val/labels, then train the HBB model.")


if __name__ == "__main__":
    main()
