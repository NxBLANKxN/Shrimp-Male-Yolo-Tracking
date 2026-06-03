# modules/analyzer.py

from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import (
    DEFAULT_KEYFRAMES,
    DEFAULT_SKIP_FRAMES,
    DEFAULT_TIME_WINDOW_SEC,
    DEFAULT_TOTAL_SHRIMP,
    HBB_CONF,
    ID_MATCH_DISTANCE,
    IMGSZ_HBB,
    IMGSZ_OBB,
    MALE_RATE_THRESHOLD,
    MIN_OBSERVATIONS_PER_SHRIMP,
    MODEL_HBB_PATH,
    MODEL_OBB_PATH,
)
from .id_assigner import FixedIDAssigner
from .preprocessing import crop_oriented_box
from .reporting import ReportWriter


class ShrimpSexRatioAnalyzer:
    """Analyze a bucket video and estimate per-shrimp sex from multiple frames."""

    def __init__(self) -> None:
        print("Loading models...")
        self.obb_model = YOLO(MODEL_OBB_PATH)
        self.hbb_model = YOLO(MODEL_HBB_PATH)
        self._preview_crop_refs: dict[int, np.ndarray] = {}

    def run(
        self,
        video_path: str,
        output_root: str = "outputs",
        total_shrimp: int = DEFAULT_TOTAL_SHRIMP,
        auto_total: bool = False,
        auto_total_percentile: float = 90.0,
        auto_total_skip_frames: int | None = None,
        auto_total_max_frames: int | None = None,
        skip_frames: int = DEFAULT_SKIP_FRAMES,
        keyframes: int = DEFAULT_KEYFRAMES,
        window_sec: int = DEFAULT_TIME_WINDOW_SEC,
        truth_csv: str | None = None,
        gt_male: int | None = None,
        gt_female: int | None = None,
        preview: bool = False,
        preview_only: bool = False,
        preview_scale: float = 0.75,
        preview_wait_ms: int = 1,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法讀取影片: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview = preview or preview_only

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        auto_total_counts: pd.DataFrame | None = None
        auto_total_summary: dict | None = None
        if auto_total:
            prescan_skip_frames = max(1, int(auto_total_skip_frames or skip_frames))
            auto_total_counts, auto_total_summary = self._estimate_total_shrimp(
                cap=cap,
                fps=fps,
                total_frames=total_frames,
                skip_frames=prescan_skip_frames,
                percentile=auto_total_percentile,
                max_frames=auto_total_max_frames,
            )
            total_shrimp = int(auto_total_summary["Recommended_Total_Shrimp"])
            print(
                "Auto-estimated total shrimp: "
                f"{total_shrimp} using P{auto_total_summary['Percentile_Used']:.0f} "
                f"(median={auto_total_summary['Median_Count']}, max={auto_total_summary['Max_Count']})"
            )
        assigner = FixedIDAssigner(total_ids=total_shrimp, match_distance=ID_MATCH_DISTANCE)

        detections: list[dict] = []
        keyframe_pool: list[dict] = []
        sample_count = max(1, (total_frames + skip_frames - 1) // skip_frames)
        self._preview_crop_refs = {}

        print(f"Video: {video_name} | {total_frames} frames | {fps:.1f} FPS")
        print(f"Sampling: every {skip_frames} frames | fixed IDs: 1..{total_shrimp}")
        if preview_only:
            print("Preview-only mode enabled. No CSV files, figures, or keyframes will be written.")
            print("Press q or Esc in the preview window to stop.")
        elif preview:
            print("Live preview enabled. Press q or Esc in the preview window to stop early.")

        frame_idx = 0
        stop_requested = False
        with tqdm(total=sample_count, desc="Analyze frames", unit="frame", ncols=85) as pbar:
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break

                frame_dets = assigner.assign(self._detect_frame(frame))
                self._stabilize_preview_crops(frame_dets)
                for det in frame_dets:
                    detections.append(self._record_detection(det, frame_idx, fps))

                if frame_dets:
                    preview_item = self._make_keyframe(frame, frame_idx, fps, frame_dets)
                    keyframe_pool.append(preview_item)
                else:
                    preview_item = self._make_empty_preview(frame, frame_idx, fps)

                if preview and self._show_preview(preview_item.get("Preview_Image", preview_item["Image"]), preview_scale, preview_wait_ms):
                    stop_requested = True

                frame_idx += skip_frames
                pbar.update(1)
                if stop_requested:
                    break

        cap.release()
        if preview:
            cv2.destroyAllWindows()

        if preview_only:
            print("\nPreview complete")
            if stop_requested:
                print("Preview stopped early by user.")
            print("No analysis outputs were written.")
            return {"summary": {}, "paths": {}, "output_dir": ""}

        det_df = pd.DataFrame(detections)
        stats_df = det_df[det_df["Include_In_Stats"] == True].copy() if "Include_In_Stats" in det_df.columns else det_df
        truth = self._load_truth(truth_csv)
        if truth and not det_df.empty:
            det_df["True_Label"] = det_df["Shrimp_ID"].map(truth).fillna("")
            stats_df["True_Label"] = stats_df["Shrimp_ID"].map(truth).fillna("")
        shrimp_df = self._summarize_shrimps(stats_df, total_shrimp)
        if truth:
            shrimp_df["True_Label"] = shrimp_df["Shrimp_ID"].map(truth).fillna("")
            shrimp_df["Correct"] = shrimp_df.apply(
                lambda r: bool(r["Final_Label"] == r["True_Label"]) if r["True_Label"] and r["Final_Label"] != "Unknown" else "",
                axis=1,
            )
        summary = self._summarize_video(shrimp_df, stats_df, video_name, fps, total_frames, gt_male, gt_female)
        summary["Total_Shrimp"] = total_shrimp
        summary["Auto_Total_Enabled"] = bool(auto_total)
        if auto_total_summary:
            summary["Auto_Total_Recommended"] = auto_total_summary["Recommended_Total_Shrimp"]
            summary["Auto_Total_Percentile"] = auto_total_summary["Percentile_Used"]
            summary["Auto_Total_Skip_Frames"] = auto_total_summary["Skip_Frames"]
            summary["Auto_Total_Max_Frames"] = auto_total_summary["Max_Frames"]
        time_windows = self._summarize_time_windows(stats_df, window_sec, total_frames, fps)
        evaluation = self._evaluate_predictions(stats_df, shrimp_df)
        error_cases = self._build_error_cases(shrimp_df)
        selected_keyframes = self._select_keyframes(keyframe_pool, keyframes)
        writer = ReportWriter(output_root, video_name, run_id)

        paths = writer.write_all(
            detections=det_df,
            per_shrimp=shrimp_df,
            summary=summary,
            time_windows=time_windows,
            evaluation=evaluation,
            error_cases=error_cases,
            keyframes=selected_keyframes,
            auto_total_counts=auto_total_counts,
            auto_total_summary=auto_total_summary,
        )

        print("\nAnalysis complete")
        if stop_requested:
            print("Preview stopped early by user; summaries are based on processed frames only.")
        print(f"Male: {summary['Pred_Male']} | Female: {summary['Pred_Female']} | Unknown: {summary['Unknown']}")
        print(f"Male ratio: {summary['Male_Ratio_Pct']}%")
        print(f"Output: {writer.run_dir}")
        return {"summary": summary, "paths": paths, "output_dir": writer.run_dir}

    def _estimate_total_shrimp(
        self,
        cap,
        fps: float,
        total_frames: int,
        skip_frames: int,
        percentile: float,
        max_frames: int | None,
    ) -> tuple[pd.DataFrame, dict]:
        percentile = min(100.0, max(50.0, float(percentile)))
        counts = []
        sample_count = max(1, (total_frames + skip_frames - 1) // skip_frames)
        if max_frames is not None and max_frames > 0:
            sample_count = min(sample_count, int(max_frames))
        print(f"Auto-total prescan: OBB-only every {skip_frames} frames | samples: up to {sample_count}")
        frame_idx = 0
        with tqdm(total=sample_count, desc="Estimate total", unit="frame", ncols=85) as pbar:
            while frame_idx < total_frames and len(counts) < sample_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                count, mean_conf = self._count_obb_shrimps(frame)
                counts.append({
                    "Frame": frame_idx,
                    "Time_Sec": round(frame_idx / fps, 2),
                    "OBB_Detection_Count": count,
                    "Mean_OBB_Conf": round(mean_conf, 4),
                })
                frame_idx += skip_frames
                pbar.update(1)

        df = pd.DataFrame(counts)
        values = df["OBB_Detection_Count"].to_numpy() if not df.empty else np.array([DEFAULT_TOTAL_SHRIMP])
        p_count = float(np.percentile(values, percentile))
        recommended = max(1, int(np.ceil(p_count)))
        median = float(np.median(values))
        max_count = int(np.max(values))
        p50 = float(np.percentile(values, 50))
        p95 = float(np.percentile(values, 95))
        stability_gap = p95 - p50
        confidence = "High" if stability_gap <= 1 else "Medium" if stability_gap <= 2 else "Low"
        summary = {
            "Frames_Used": int(len(values)),
            "Skip_Frames": int(skip_frames),
            "Max_Frames": int(max_frames) if max_frames is not None and max_frames > 0 else "",
            "Percentile_Used": percentile,
            "Recommended_Total_Shrimp": recommended,
            "Percentile_Count": round(p_count, 2),
            "Median_Count": round(median, 2),
            "Max_Count": max_count,
            "P50_Count": round(p50, 2),
            "P95_Count": round(p95, 2),
            "Stability_Gap_P95_P50": round(stability_gap, 2),
            "Auto_Total_Confidence": confidence,
        }
        return df, summary

    def _count_obb_shrimps(self, frame) -> tuple[int, float]:
        result = self.obb_model(frame, conf=0.6, iou=0.4, imgsz=IMGSZ_OBB, verbose=False)[0]
        if result.obb is None:
            return 0, 0.0
        confs = []
        for obb in result.obb:
            cls_idx = int(obb.cls.cpu().numpy()[0])
            if result.names[cls_idx] == "shrimp":
                confs.append(float(obb.conf.cpu().numpy()[0]))
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return len(confs), mean_conf

    def _detect_frame(self, frame) -> list[dict]:
        obb_result = self.obb_model(frame, conf=0.6, iou=0.4, imgsz=IMGSZ_OBB, verbose=False)[0]
        if obb_result.obb is None:
            return []

        crops, metas = [], []
        for obb in obb_result.obb:
            cls_idx = int(obb.cls.cpu().numpy()[0])
            if obb_result.names[cls_idx] != "shrimp":
                continue
            obb_data = obb.xywhr.cpu().numpy()[0]
            try:
                crop, vertices, inverse_matrix = crop_oriented_box(frame, obb_data)
            except Exception:
                continue
            if crop.size == 0:
                continue
            crops.append(crop)
            metas.append({
                "cx": float(obb_data[0]),
                "cy": float(obb_data[1]),
                "vertices": np.asarray(vertices, dtype=np.int32).reshape(-1, 2),
                "inverse_matrix": inverse_matrix,
                "crop": crop,
            })

        if not crops:
            return []

        hbb_results = self.hbb_model(crops, imgsz=IMGSZ_HBB, verbose=False)
        detections = []
        for meta, res in zip(metas, hbb_results):
            male_conf = 0.0
            male_line_pts = None
            male_line_box_crop = None
            if getattr(res, "boxes", None) is not None:
                for box in res.boxes:
                    cls_idx = int(box.cls[0])
                    name = res.names[cls_idx].lower()
                    conf = float(box.conf[0])
                    if name == "male_line" and conf >= HBB_CONF:
                        if conf > male_conf:
                            male_conf = conf
                            male_line_pts = self._project_hbb_box(box, meta["inverse_matrix"])
                            male_line_box_crop = box.xyxy[0].cpu().numpy().astype(float).tolist()
            detections.append({
                **meta,
                "is_male": male_conf > 0,
                "male_conf": male_conf,
                "male_line_pts": male_line_pts,
                "male_line_box_crop": male_line_box_crop,
            })
        return detections

    @staticmethod
    def _project_hbb_box(box, inverse_matrix) -> np.ndarray:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        pts = np.array(
            [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
            dtype=np.float32,
        )
        return cv2.perspectiveTransform(pts, inverse_matrix).astype(np.int32).reshape(-1, 2)

    def _stabilize_preview_crops(self, detections: list[dict]) -> None:
        for det in detections:
            if not det.get("include_in_stats", True):
                continue
            shrimp_id = det.get("shrimp_id")
            if shrimp_id == "" or shrimp_id is None:
                continue
            crop = det.get("crop")
            if crop is None or getattr(crop, "size", 0) == 0:
                continue

            ref = self._preview_crop_refs.get(int(shrimp_id))
            if ref is not None and self._is_horizontal_flip_more_stable(crop, ref):
                det["crop"] = cv2.flip(crop, 1)
                det["male_line_box_crop"] = self._flip_crop_box(det.get("male_line_box_crop"), crop.shape[1])
                crop = det["crop"]

            self._preview_crop_refs[int(shrimp_id)] = self._crop_signature(crop)

    @staticmethod
    def _crop_signature(crop) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (96, 48), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _is_horizontal_flip_more_stable(crop, reference: np.ndarray) -> bool:
        current = ShrimpSexRatioAnalyzer._crop_signature(crop)
        flipped = cv2.flip(current, 1)
        current_diff = np.mean(cv2.absdiff(current, reference))
        flipped_diff = np.mean(cv2.absdiff(flipped, reference))
        return flipped_diff + 2.0 < current_diff

    @staticmethod
    def _flip_crop_box(box, crop_width: int):
        if box is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in box]
        return [crop_width - x2, y1, crop_width - x1, y2]

    @staticmethod
    def _record_detection(det: dict, frame_idx: int, fps: float) -> dict:
        vertices = det["vertices"]
        x1, y1 = vertices.min(axis=0)
        x2, y2 = vertices.max(axis=0)
        shrimp_id = det.get("shrimp_id", "")
        return {
            "Frame": frame_idx,
            "Time_Sec": round(frame_idx / fps, 2),
            "Shrimp_ID": int(shrimp_id) if shrimp_id != "" else "",
            "Pred_Label": "Male" if det["is_male"] else "Female",
            "Male_Conf": round(float(det["male_conf"]), 4),
            "ID_Status": det["id_status"],
            "ID_Distance": det["id_distance"],
            "Include_In_Stats": bool(det.get("include_in_stats", True)),
            "Center_X": round(float(det["cx"]), 1),
            "Center_Y": round(float(det["cy"]), 1),
            "Box_X1": int(x1),
            "Box_Y1": int(y1),
            "Box_X2": int(x2),
            "Box_Y2": int(y2),
        }

    def _summarize_shrimps(self, det_df: pd.DataFrame, total_shrimp: int) -> pd.DataFrame:
        rows = []
        for shrimp_id in range(1, total_shrimp + 1):
            group = det_df[det_df["Shrimp_ID"] == shrimp_id] if not det_df.empty else pd.DataFrame()
            seen = int(len(group))
            male_hits = int((group["Pred_Label"] == "Male").sum()) if seen else 0
            forced = int((group["ID_Status"] == "forced").sum()) if seen else 0
            male_rate = male_hits / seen if seen else 0.0
            enough = seen >= MIN_OBSERVATIONS_PER_SHRIMP
            final = "Unknown"
            if enough:
                final = "Male" if male_rate >= MALE_RATE_THRESHOLD else "Female"
            rows.append({
                "Shrimp_ID": shrimp_id,
                "Total_Seen": seen,
                "Male_Hits": male_hits,
                "Female_Hits": seen - male_hits,
                "Male_Rate_Pct": round(male_rate * 100, 2),
                "Mean_Male_Conf": round(float(group["Male_Conf"].mean()), 4) if seen else 0.0,
                "Forced_ID_Count": forced,
                "Forced_ID_Rate_Pct": round(forced / seen * 100, 2) if seen else 0.0,
                "Decision_Margin_Pct": round(abs(male_rate - MALE_RATE_THRESHOLD) * 100, 2),
                "Final_Label": final,
                "Enough_Evidence": enough,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _load_truth(path: str | None) -> dict[int, str]:
        if not path:
            return {}
        df = pd.read_csv(path)
        required = {"Shrimp_ID", "True_Label"}
        if not required.issubset(df.columns):
            raise ValueError("truth CSV must contain Shrimp_ID and True_Label columns")
        truth = {}
        for _, row in df.iterrows():
            label = str(row["True_Label"]).strip().capitalize()
            if label in {"M", "Male"}:
                label = "Male"
            elif label in {"F", "Female"}:
                label = "Female"
            else:
                continue
            truth[int(row["Shrimp_ID"])] = label
        return truth

    @staticmethod
    def _summarize_time_windows(
        det_df: pd.DataFrame,
        window_sec: int,
        total_frames: int,
        fps: float,
    ) -> pd.DataFrame:
        window_sec = max(window_sec, 1)
        duration_sec = total_frames / max(fps, 1.0)
        total_windows = max(1, int(np.ceil(duration_sec / window_sec)))
        df = det_df.copy()
        if not df.empty:
            df["Window_Index"] = (df["Time_Sec"] // window_sec).astype(int)
        rows = []
        for win in range(total_windows):
            group = df[df["Window_Index"] == win] if not df.empty else pd.DataFrame()
            if group.empty:
                observed_ids = male = female = detections = 0
                forced_rate = 0.0
            else:
                per_id = group.groupby("Shrimp_ID")["Pred_Label"].agg(
                    lambda s: "Male" if (s == "Male").mean() >= MALE_RATE_THRESHOLD else "Female"
                )
                male = int((per_id == "Male").sum())
                female = int((per_id == "Female").sum())
                observed_ids = int(per_id.size)
                detections = int(len(group))
                forced_rate = round((group["ID_Status"] == "forced").mean() * 100, 2)
            total = male + female
            rows.append({
                "Window_Index": win,
                "Start_Sec": int(win * window_sec),
                "End_Sec": round(min((win + 1) * window_sec, duration_sec), 2),
                "Start_Frame": int(round(win * window_sec * fps)),
                "End_Frame": min(int(round((win + 1) * window_sec * fps)) - 1, total_frames - 1),
                "Observed_IDs": observed_ids,
                "Male": male,
                "Female": female,
                "Male_Ratio_Pct": round(male / total * 100, 2) if total else 0,
                "Detections": detections,
                "Forced_ID_Rate_Pct": forced_rate,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _evaluate_predictions(det_df: pd.DataFrame, shrimp_df: pd.DataFrame) -> dict:
        result = {}
        reliable = shrimp_df[shrimp_df["Enough_Evidence"] == True]
        result["Reliable_Shrimp_Count"] = int(len(reliable))
        result["Mean_Decision_Margin_Pct"] = round(float(shrimp_df["Decision_Margin_Pct"].mean()), 2)
        result["Mean_Forced_ID_Rate_Pct"] = round(float(shrimp_df["Forced_ID_Rate_Pct"].mean()), 2)
        result["Min_Observations_Per_Shrimp"] = int(shrimp_df["Total_Seen"].min()) if not shrimp_df.empty else 0

        coverage = min(1.0, result["Reliable_Shrimp_Count"] / max(len(shrimp_df), 1))
        margin = min(1.0, result["Mean_Decision_Margin_Pct"] / 40.0)
        id_quality = max(0.0, 1.0 - result["Mean_Forced_ID_Rate_Pct"] / 100.0)
        result["Reliability_Score"] = round((coverage * 0.45 + margin * 0.30 + id_quality * 0.25) * 100, 2)

        if "True_Label" in shrimp_df.columns and shrimp_df["True_Label"].astype(bool).any():
            valid = shrimp_df[(shrimp_df["True_Label"] != "") & (shrimp_df["Final_Label"] != "Unknown")]
            result["Multi_Frame_Accuracy_Pct"] = round((valid["True_Label"] == valid["Final_Label"]).mean() * 100, 2) if len(valid) else ""
            det_valid = det_df[det_df.get("True_Label", "") != ""] if "True_Label" in det_df.columns else pd.DataFrame()
            result["Single_Frame_Accuracy_Pct"] = round((det_valid["True_Label"] == det_valid["Pred_Label"]).mean() * 100, 2) if len(det_valid) else ""
        return result

    @staticmethod
    def _build_error_cases(shrimp_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in shrimp_df.iterrows():
            reasons = []
            if not row["Enough_Evidence"]:
                reasons.append("low_observation")
            if row["Forced_ID_Rate_Pct"] > 35:
                reasons.append("high_forced_id")
            if row["Decision_Margin_Pct"] < 10:
                reasons.append("near_threshold")
            if "Correct" in row and row["Correct"] == False:
                reasons.append("wrong_prediction")
            if reasons:
                rows.append({**row.to_dict(), "Reasons": ";".join(reasons)})
        return pd.DataFrame(rows)

    @staticmethod
    def _summarize_video(
        shrimp_df: pd.DataFrame,
        det_df: pd.DataFrame,
        video_name: str,
        fps: float,
        total_frames: int,
        gt_male: int | None,
        gt_female: int | None,
    ) -> dict:
        pred_male = int((shrimp_df["Final_Label"] == "Male").sum())
        pred_female = int((shrimp_df["Final_Label"] == "Female").sum())
        unknown = int((shrimp_df["Final_Label"] == "Unknown").sum())
        decided = max(pred_male + pred_female, 1)

        summary = {
            "Video": video_name,
            "OBB_Model": MODEL_OBB_PATH,
            "HBB_Model": MODEL_HBB_PATH,
            "IMGSZ_OBB": IMGSZ_OBB,
            "IMGSZ_HBB": IMGSZ_HBB,
            "HBB_CONF": HBB_CONF,
            "MALE_RATE_THRESHOLD": MALE_RATE_THRESHOLD,
            "MIN_OBSERVATIONS_PER_SHRIMP": MIN_OBSERVATIONS_PER_SHRIMP,
            "Duration_Sec": round(total_frames / fps, 2),
            "Sampled_Detections": int(len(det_df)),
            "Pred_Male": pred_male,
            "Pred_Female": pred_female,
            "Unknown": unknown,
            "Male_Ratio_Pct": round(pred_male / decided * 100, 2),
            "Female_Ratio_Pct": round(pred_female / decided * 100, 2),
            "Mean_Observations_Per_Shrimp": round(float(shrimp_df["Total_Seen"].mean()), 2),
            "Reliable_Shrimp_Count": int(shrimp_df["Enough_Evidence"].sum()),
            "Mean_Forced_ID_Rate_Pct": round(float(shrimp_df["Forced_ID_Rate_Pct"].mean()), 2),
            "GT_Male": gt_male if gt_male is not None else "",
            "GT_Female": gt_female if gt_female is not None else "",
        }
        if gt_male is not None and gt_female is not None:
            male_acc = max(0.0, 1.0 - abs(pred_male - gt_male) / max(gt_male, 1))
            female_acc = max(0.0, 1.0 - abs(pred_female - gt_female) / max(gt_female, 1))
            summary["Count_Accuracy_Pct"] = round((male_acc + female_acc) / 2 * 100, 2)
            summary["Exact_Count_Match"] = pred_male == gt_male and pred_female == gt_female
        return summary

    @staticmethod
    def _make_keyframe(frame, frame_idx: int, fps: float, detections: list[dict]) -> dict:
        annotated = frame.copy()
        male_count = 0
        for det in detections:
            if not det.get("include_in_stats", True):
                color = (150, 150, 150)
                label = "OVF"
                cv2.polylines(annotated, [det["vertices"]], True, color, 1)
                x, y = det["vertices"][0]
                cv2.putText(annotated, label, (int(x), max(24, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                continue
            is_male = det["is_male"]
            male_count += int(is_male)
            color = (220, 110, 40) if is_male else (80, 190, 90)
            label = f"ID{det['shrimp_id']} M {det['male_conf']:.2f}" if is_male else f"ID{det['shrimp_id']} F"
            cv2.polylines(annotated, [det["vertices"]], True, color, 2)
            if det.get("male_line_pts") is not None:
                cv2.polylines(annotated, [det["male_line_pts"]], True, (0, 255, 255), 2)
            x, y = det["vertices"][0]
            cv2.putText(annotated, label, (int(x), max(24, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.rectangle(annotated, (8, 8), (460, 74), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            f"Frame {frame_idx} | {frame_idx / fps:.1f}s | M {male_count} F {len(detections)-male_count}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
        preview_image = ShrimpSexRatioAnalyzer._compose_preview_canvas(annotated, detections)
        return {
            "Frame": frame_idx,
            "Time_Sec": round(frame_idx / fps, 2),
            "Detected_Total": len(detections),
            "Detected_Male": male_count,
            "Detected_Female": len(detections) - male_count,
            "Score": len(detections),
            "Image": annotated,
            "Preview_Image": preview_image,
            "Image_Path": "",
        }

    @staticmethod
    def _make_empty_preview(frame, frame_idx: int, fps: float) -> dict:
        annotated = frame.copy()
        cv2.rectangle(annotated, (8, 8), (520, 74), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            f"Frame {frame_idx} | {frame_idx / fps:.1f}s | no shrimp detections",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
        cv2.putText(annotated, "Live preview: press q or Esc to stop", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
        return {
            "Frame": frame_idx,
            "Time_Sec": round(frame_idx / fps, 2),
            "Detected_Total": 0,
            "Detected_Male": 0,
            "Detected_Female": 0,
            "Score": 0,
            "Image": annotated,
            "Preview_Image": annotated,
            "Image_Path": "",
        }

    @staticmethod
    def _compose_preview_canvas(annotated, detections: list[dict]):
        detections = sorted(detections, key=ShrimpSexRatioAnalyzer._preview_sort_key)
        panel_width = 620
        columns = 1
        margin = 10
        tile_w = (panel_width - margin * (columns + 1)) // columns
        tile_h = 150
        header_h = 38
        rows = max(1, int(np.ceil(len(detections) / columns)))
        height = max(annotated.shape[0], header_h + rows * (tile_h + margin) + margin)
        canvas = np.full((height, annotated.shape[1] + panel_width, 3), 245, dtype=np.uint8)
        canvas[: annotated.shape[0], : annotated.shape[1]] = annotated

        x0 = annotated.shape[1]
        cv2.rectangle(canvas, (x0, 0), (x0 + panel_width, height), (235, 235, 235), -1)
        cv2.putText(canvas, "DEBUG Window", (x0 + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2)

        for idx, det in enumerate(detections, start=1):
            row = (idx - 1) // columns
            col = (idx - 1) % columns
            y = header_h + row * (tile_h + margin)
            x = x0 + margin + col * (tile_w + margin)
            if y + tile_h > height:
                break
            tile = ShrimpSexRatioAnalyzer._crop_preview_tile(det, tile_w, tile_h, idx)
            canvas[y : y + tile_h, x : x + tile.shape[1]] = tile
        return canvas

    @staticmethod
    def _preview_sort_key(det: dict):
        if not det.get("include_in_stats", True):
            return (1, 9999)
        shrimp_id = det.get("shrimp_id", 9999)
        try:
            shrimp_id = int(shrimp_id)
        except (TypeError, ValueError):
            shrimp_id = 9999
        return (0, shrimp_id)

    @staticmethod
    def _crop_preview_tile(det: dict, width: int, height: int, idx: int):
        tile = np.full((height, width, 3), 245, dtype=np.uint8)
        crop = det.get("crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            return tile

        draw_crop, male_line_box = ShrimpSexRatioAnalyzer._trim_preview_crop(crop, det.get("male_line_box_crop"))
        if male_line_box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in male_line_box]
            cv2.rectangle(draw_crop, (x1, y1), (x2, y2), (0, 255, 255), 3)

        label_h = 22
        image_h = height - label_h
        scale = width / max(draw_crop.shape[1], 1)
        if draw_crop.shape[0] * scale > image_h:
            scale = image_h / max(draw_crop.shape[0], 1)
        resized = cv2.resize(
            draw_crop,
            (max(1, int(draw_crop.shape[1] * scale)), max(1, int(draw_crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        y0 = 2
        x0 = max(0, (width - resized.shape[1]) // 2)
        tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized

        if not det.get("include_in_stats", True):
            label = f"{idx}. OVF"
            color = (110, 110, 110)
        else:
            sex = "M" if det.get("is_male") else "F"
            shrimp_id = det.get("shrimp_id", "")
            conf = float(det.get("male_conf", 0.0))
            label = f"{idx}. ID{shrimp_id} {sex}" + (f" {conf:.2f}" if sex == "M" else "")
            color = (220, 110, 40) if sex == "M" else (80, 160, 80)
        overlay = tile.copy()
        cv2.rectangle(overlay, (0, height - label_h), (width, height), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.82, tile, 0.18, 0, tile)
        cv2.putText(tile, label, (8, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), color, 1)
        return tile

    @staticmethod
    def _trim_preview_crop(crop, male_line_box):
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return crop.copy(), male_line_box

        border = np.concatenate(
            [
                crop[: max(1, h // 20), :, :].reshape(-1, 3),
                crop[-max(1, h // 20) :, :, :].reshape(-1, 3),
                crop[:, : max(1, w // 20), :].reshape(-1, 3),
                crop[:, -max(1, w // 20) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(border.astype(np.float32), axis=0)
        diff = np.linalg.norm(crop.astype(np.float32) - bg, axis=2)
        mask = diff > 18
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return crop.copy(), male_line_box

        pad_x = max(4, int(w * 0.04))
        pad_y = max(4, int(h * 0.04))
        y1 = max(0, int(rows.min()) - pad_y)
        y2 = min(h, int(rows.max()) + pad_y + 1)
        x1 = max(0, int(cols.min()) - pad_x)
        x2 = min(w, int(cols.max()) + pad_x + 1)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            return crop.copy(), male_line_box

        trimmed = crop[y1:y2, x1:x2].copy()
        adjusted_box = None
        if male_line_box is not None:
            bx1, by1, bx2, by2 = [float(v) for v in male_line_box]
            adjusted_box = [
                max(0.0, bx1 - x1),
                max(0.0, by1 - y1),
                min(float(x2 - x1 - 1), bx2 - x1),
                min(float(y2 - y1 - 1), by2 - y1),
            ]
        return trimmed, adjusted_box

    @staticmethod
    def _show_preview(image, scale: float, wait_ms: int) -> bool:
        scale = max(float(scale), 0.05)
        wait_ms = max(int(wait_ms), 1)
        display = image
        if scale != 1.0:
            display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Shrimp Analysis Preview", display)
        key = cv2.waitKey(wait_ms) & 0xFF
        return key in (27, ord("q"), ord("Q"))

    @staticmethod
    def _select_keyframes(keyframes: list[dict], count: int) -> list[dict]:
        return sorted(keyframes, key=lambda k: k["Score"], reverse=True)[: max(0, count)]
