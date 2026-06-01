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

    def run(
        self,
        video_path: str,
        output_root: str = "outputs",
        total_shrimp: int = DEFAULT_TOTAL_SHRIMP,
        skip_frames: int = DEFAULT_SKIP_FRAMES,
        keyframes: int = DEFAULT_KEYFRAMES,
        window_sec: int = DEFAULT_TIME_WINDOW_SEC,
        truth_csv: str | None = None,
        gt_male: int | None = None,
        gt_female: int | None = None,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法讀取影片: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = ReportWriter(output_root, video_name, run_id)

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assigner = FixedIDAssigner(total_ids=total_shrimp, match_distance=ID_MATCH_DISTANCE)

        detections: list[dict] = []
        keyframe_pool: list[dict] = []
        sample_count = max(1, (total_frames + skip_frames - 1) // skip_frames)

        print(f"Video: {video_name} | {total_frames} frames | {fps:.1f} FPS")
        print(f"Sampling: every {skip_frames} frames | fixed IDs: 1..{total_shrimp}")

        frame_idx = 0
        with tqdm(total=sample_count, desc="Analyze frames", unit="frame", ncols=85) as pbar:
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break

                frame_dets = assigner.assign(self._detect_frame(frame))
                for det in frame_dets:
                    detections.append(self._record_detection(det, frame_idx, fps))

                if frame_dets:
                    keyframe_pool.append(self._make_keyframe(frame, frame_idx, fps, frame_dets))

                frame_idx += skip_frames
                pbar.update(1)

        cap.release()

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
        time_windows = self._summarize_time_windows(stats_df, window_sec, total_frames, fps)
        evaluation = self._evaluate_predictions(stats_df, shrimp_df)
        error_cases = self._build_error_cases(shrimp_df)
        selected_keyframes = self._select_keyframes(keyframe_pool, keyframes)

        paths = writer.write_all(
            detections=det_df,
            per_shrimp=shrimp_df,
            summary=summary,
            time_windows=time_windows,
            evaluation=evaluation,
            error_cases=error_cases,
            keyframes=selected_keyframes,
        )

        print("\nAnalysis complete")
        print(f"Male: {summary['Pred_Male']} | Female: {summary['Pred_Female']} | Unknown: {summary['Unknown']}")
        print(f"Male ratio: {summary['Male_Ratio_Pct']}%")
        print(f"Output: {writer.run_dir}")
        return {"summary": summary, "paths": paths, "output_dir": writer.run_dir}

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
            })

        if not crops:
            return []

        hbb_results = self.hbb_model(crops, imgsz=IMGSZ_HBB, verbose=False)
        detections = []
        for meta, res in zip(metas, hbb_results):
            male_conf = 0.0
            male_line_pts = None
            if getattr(res, "boxes", None) is not None:
                for box in res.boxes:
                    cls_idx = int(box.cls[0])
                    name = res.names[cls_idx].lower()
                    conf = float(box.conf[0])
                    if name == "male_line" and conf >= HBB_CONF:
                        if conf > male_conf:
                            male_conf = conf
                            male_line_pts = self._project_hbb_box(box, meta["inverse_matrix"])
            detections.append({
                **meta,
                "is_male": male_conf > 0,
                "male_conf": male_conf,
                "male_line_pts": male_line_pts,
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
        cv2.putText(annotated, "Review evidence frame", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
        return {
            "Frame": frame_idx,
            "Time_Sec": round(frame_idx / fps, 2),
            "Detected_Total": len(detections),
            "Detected_Male": male_count,
            "Detected_Female": len(detections) - male_count,
            "Score": len(detections),
            "Image": annotated,
            "Image_Path": "",
        }

    @staticmethod
    def _select_keyframes(keyframes: list[dict], count: int) -> list[dict]:
        return sorted(keyframes, key=lambda k: k["Score"], reverse=True)[: max(0, count)]
