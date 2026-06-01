# Fast-Stats.py

import argparse

from modules.analyzer import ShrimpSexRatioAnalyzer
from modules.config import DEFAULT_KEYFRAMES, DEFAULT_SKIP_FRAMES, DEFAULT_TIME_WINDOW_SEC, DEFAULT_TOTAL_SHRIMP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze shrimp sex ratio from a bottom-view bucket video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-root", default="outputs", help="Output root directory.")
    parser.add_argument("--total-shrimp", type=int, default=DEFAULT_TOTAL_SHRIMP, help="Expected shrimp count in the bucket.")
    parser.add_argument("--skip-frames", type=int, default=DEFAULT_SKIP_FRAMES, help="Analyze every N frames.")
    parser.add_argument("--keyframes", type=int, default=DEFAULT_KEYFRAMES, help="Number of evidence keyframes to export.")
    parser.add_argument("--window-sec", type=int, default=DEFAULT_TIME_WINDOW_SEC, help="Seconds per temporal summary window.")
    parser.add_argument("--truth-csv", help="Optional CSV with columns Shrimp_ID,True_Label for evaluation.")
    parser.add_argument("--gt-male", type=int, help="Optional ground-truth male count for evaluation.")
    parser.add_argument("--gt-female", type=int, help="Optional ground-truth female count for evaluation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyzer = ShrimpSexRatioAnalyzer()
    analyzer.run(
        video_path=args.video,
        output_root=args.output_root,
        total_shrimp=args.total_shrimp,
        skip_frames=args.skip_frames,
        keyframes=args.keyframes,
        window_sec=args.window_sec,
        truth_csv=args.truth_csv,
        gt_male=args.gt_male,
        gt_female=args.gt_female,
    )


if __name__ == "__main__":
    main()
