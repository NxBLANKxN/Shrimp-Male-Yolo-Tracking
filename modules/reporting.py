# modules/reporting.py

from __future__ import annotations

import os

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config import MALE_RATE_THRESHOLD, MIN_OBSERVATIONS_PER_SHRIMP


plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


class ReportWriter:
    def __init__(self, output_root: str, video_name: str, run_id: str) -> None:
        self.run_dir = os.path.abspath(os.path.join(output_root, video_name, f"analysis_{run_id}"))
        self.data_dir = os.path.join(self.run_dir, "data")
        self.figure_dir = os.path.join(self.run_dir, "figures")
        self.keyframe_dir = os.path.join(self.run_dir, "evidence", "keyframes")
        for path in (self.data_dir, self.figure_dir, self.keyframe_dir):
            os.makedirs(path, exist_ok=True)

    def write_all(
        self,
        detections: pd.DataFrame,
        per_shrimp: pd.DataFrame,
        summary: dict,
        time_windows: pd.DataFrame,
        evaluation: dict,
        error_cases: pd.DataFrame,
        keyframes: list[dict],
        auto_total_counts: pd.DataFrame | None = None,
        auto_total_summary: dict | None = None,
    ) -> dict:
        paths = {}
        paths["detections"] = self._write_csv(detections, "detections.csv")
        paths["per_shrimp"] = self._write_csv(per_shrimp, "per_shrimp_summary.csv")
        paths["truth_template"] = self._write_truth_template(per_shrimp)
        paths["summary"] = self._write_csv(pd.DataFrame([summary]), "bucket_summary.csv")
        paths["time_windows"] = self._write_csv(time_windows, "time_window_summary.csv")
        paths["evaluation"] = self._write_csv(pd.DataFrame([evaluation]), "evaluation_summary.csv")
        paths["error_cases"] = self._write_csv(error_cases, "error_cases.csv")
        paths["keyframes"] = self._write_keyframes(keyframes)
        paths["keyframe_index"] = self._write_csv(pd.DataFrame(keyframes).drop(columns=["Image"], errors="ignore"), "keyframe_index.csv")
        reliability = self._build_reliability_summary(detections, per_shrimp, time_windows, keyframes)
        paths["reliability_summary"] = self._write_csv(pd.DataFrame([reliability]), "reliability_summary.csv")
        paths["ratio_chart"] = self._plot_ratio(summary)
        paths["shrimp_chart"] = self._plot_per_shrimp(per_shrimp)
        paths["temporal_chart"] = self._plot_temporal(detections, per_shrimp, time_windows)
        if auto_total_counts is not None and auto_total_summary is not None:
            paths["auto_total_counts"] = self._write_csv(auto_total_counts, "auto_total_counts.csv")
            paths["auto_total_summary"] = self._write_csv(pd.DataFrame([auto_total_summary]), "auto_total_summary.csv")
            paths["auto_total_chart"] = self._plot_auto_total(auto_total_counts, auto_total_summary)
        if evaluation.get("Single_Frame_Accuracy_Pct", "") != "":
            paths["accuracy_chart"] = self._plot_single_vs_multi(evaluation)
            paths["confusion_matrix"] = self._write_confusion_matrix(per_shrimp)
            paths["confusion_chart"] = self._plot_confusion_matrix(per_shrimp)
        return paths

    def _write_csv(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.data_dir, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _write_truth_template(self, per_shrimp: pd.DataFrame) -> str:
        df = per_shrimp[["Shrimp_ID", "Final_Label", "Total_Seen", "Male_Rate_Pct"]].copy()
        df["True_Label"] = ""
        df["Notes"] = ""
        path = os.path.join(self.data_dir, "truth_template.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    @staticmethod
    def _safe_pct(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return round(float(numerator) / float(denominator) * 100, 2)

    def _build_reliability_summary(
        self,
        detections: pd.DataFrame,
        per_shrimp: pd.DataFrame,
        time_windows: pd.DataFrame,
        keyframes: list[dict],
    ) -> dict:
        total_shrimp = int(len(per_shrimp))
        analyzed_frames = int(detections["Frame"].nunique()) if not detections.empty and "Frame" in detections.columns else 0
        included = detections[detections["Include_In_Stats"] == True].copy() if "Include_In_Stats" in detections.columns else detections.copy()
        overflow = detections[detections["ID_Status"] == "overflow"].copy() if "ID_Status" in detections.columns else pd.DataFrame()
        forced = included[included["ID_Status"] == "forced"].copy() if "ID_Status" in included.columns else pd.DataFrame()

        observed_ids = int((per_shrimp["Total_Seen"] > 0).sum()) if "Total_Seen" in per_shrimp.columns else 0
        enough_ids = int((per_shrimp["Enough_Evidence"] == True).sum()) if "Enough_Evidence" in per_shrimp.columns else 0
        male_line_hits = int((included["Male_Conf"] > 0).sum()) if "Male_Conf" in included.columns and not included.empty else 0

        id_distance = pd.to_numeric(included.get("ID_Distance", pd.Series(dtype=float)), errors="coerce")
        male_ratio = pd.to_numeric(time_windows.get("Male_Ratio_Pct", pd.Series(dtype=float)), errors="coerce")
        nonempty_windows = time_windows[time_windows["Detections"] > 0] if "Detections" in time_windows.columns else time_windows

        return {
            "Total_Shrimp": total_shrimp,
            "Analyzed_Frames": analyzed_frames,
            "Included_Detections": int(len(included)),
            "Overflow_Detections": int(len(overflow)),
            "Overflow_Rate_Pct": self._safe_pct(len(overflow), len(detections)),
            "Frames_With_Overflow": int(overflow["Frame"].nunique()) if not overflow.empty and "Frame" in overflow.columns else 0,
            "Overflow_Frame_Rate_Pct": self._safe_pct(
                overflow["Frame"].nunique() if not overflow.empty and "Frame" in overflow.columns else 0,
                analyzed_frames,
            ),
            "Observed_IDs": observed_ids,
            "ID_Coverage_Rate_Pct": self._safe_pct(observed_ids, total_shrimp),
            "Enough_Evidence_IDs": enough_ids,
            "Enough_Evidence_Rate_Pct": self._safe_pct(enough_ids, total_shrimp),
            "Mean_Observations_Per_ID": round(float(per_shrimp["Total_Seen"].mean()), 2) if "Total_Seen" in per_shrimp.columns else 0.0,
            "Min_Observations_Per_ID": int(per_shrimp["Total_Seen"].min()) if "Total_Seen" in per_shrimp.columns and not per_shrimp.empty else 0,
            "Mean_Forced_ID_Rate_Pct": round(float(per_shrimp["Forced_ID_Rate_Pct"].mean()), 2) if "Forced_ID_Rate_Pct" in per_shrimp.columns else 0.0,
            "Max_Forced_ID_Rate_Pct": round(float(per_shrimp["Forced_ID_Rate_Pct"].max()), 2) if "Forced_ID_Rate_Pct" in per_shrimp.columns else 0.0,
            "Forced_Detections": int(len(forced)),
            "Forced_Detection_Rate_Pct": self._safe_pct(len(forced), len(included)),
            "Mean_ID_Distance": round(float(id_distance.mean()), 2) if not id_distance.dropna().empty else 0.0,
            "Max_ID_Distance": round(float(id_distance.max()), 2) if not id_distance.dropna().empty else 0.0,
            "Male_Line_Detection_Count": male_line_hits,
            "Male_Line_Detection_Rate_Pct": self._safe_pct(male_line_hits, len(included)),
            "Mean_Male_Conf": round(float(included["Male_Conf"].mean()), 4) if "Male_Conf" in included.columns and not included.empty else 0.0,
            "Time_Window_Male_Ratio_Std": round(float(male_ratio.std()), 2) if len(male_ratio.dropna()) > 1 else 0.0,
            "Time_Window_Male_Ratio_Range": round(float(male_ratio.max() - male_ratio.min()), 2) if not male_ratio.dropna().empty else 0.0,
            "Mean_Observed_IDs_Per_Window": round(float(nonempty_windows["Observed_IDs"].mean()), 2) if "Observed_IDs" in nonempty_windows.columns and not nonempty_windows.empty else 0.0,
            "Keyframe_Count": int(len(keyframes)),
        }

    def _write_keyframes(self, keyframes: list[dict]) -> list[str]:
        paths = []
        for idx, item in enumerate(keyframes, start=1):
            path = os.path.join(self.keyframe_dir, f"keyframe_{idx:02d}_frame_{item['Frame']:06d}.jpg")
            cv2.imwrite(path, item["Image"])
            item["Image_Path"] = path
            paths.append(path)
        return paths

    def _plot_ratio(self, summary: dict) -> str:
        path = os.path.join(self.figure_dir, "sex_ratio_summary.png")
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        labels = ["公蝦", "母蝦", "未定"]
        values = [summary["Pred_Male"], summary["Pred_Female"], summary["Unknown"]]
        colors = ["#2F6FDB", "#D94F70", "#8A8F98"]
        ax.bar(labels, values, color=colors)
        ax.set_title("草蝦公母數量估計", fontsize=14, fontweight="bold")
        ax.set_ylabel("數量")
        for i, value in enumerate(values):
            ax.text(i, value + 0.05, str(value), ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def _plot_per_shrimp(self, per_shrimp: pd.DataFrame) -> str:
        path = os.path.join(self.figure_dir, "per_shrimp_male_rate.png")
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        colors = ["#2F6FDB" if v == "Male" else "#D94F70" if v == "Female" else "#8A8F98" for v in per_shrimp["Final_Label"]]
        x = [f"ID{sid}" for sid in per_shrimp["Shrimp_ID"]]
        ax.bar(x, per_shrimp["Male_Rate_Pct"], color=colors)
        threshold_pct = MALE_RATE_THRESHOLD * 100
        ax.axhline(threshold_pct, color="#222222", linestyle="--", label=f"公蝦判定門檻 {threshold_pct:.0f}%")
        ax.set_ylim(0, 112)
        ax.set_title("每隻蝦為公蝦的概率", fontsize=14, fontweight="bold")
        ax.set_xlabel("系統分配 ID")
        ax.set_ylabel("為公蝦的概率 (%)")
        ax.legend()
        for i, row in per_shrimp.iterrows():
            ax.text(
                i,
                min(float(row["Male_Rate_Pct"]) + 2, 106),
                f"male_line={int(row['Male_Hits'])}\nn={int(row['Total_Seen'])}",
                ha="center",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def _plot_temporal(self, detections: pd.DataFrame, per_shrimp: pd.DataFrame, time_windows: pd.DataFrame) -> str:
        path = os.path.join(self.figure_dir, "temporal_sex_ratio.png")
        shrimp_ids = per_shrimp["Shrimp_ID"].astype(int).tolist() if "Shrimp_ID" in per_shrimp.columns else []
        fig_w = max(9, len(time_windows) * 0.75)
        fig_h = max(4.8, len(shrimp_ids) * 0.65 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

        if time_windows.empty or not shrimp_ids:
            ax.set_title("各 ID 時間窗公母變化", fontsize=14, fontweight="bold")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return path

        matrix = np.zeros((len(shrimp_ids), len(time_windows)), dtype=int)
        labels = [["-" for _ in range(len(time_windows))] for _ in shrimp_ids]
        df = detections.copy()
        if not df.empty and "Include_In_Stats" in df.columns:
            df = df[df["Include_In_Stats"] == True].copy()

        for col_idx, (_, window) in enumerate(time_windows.iterrows()):
            start_sec = float(window["Start_Sec"])
            end_sec = float(window["End_Sec"])
            if df.empty:
                window_df = pd.DataFrame()
            else:
                window_df = df[(df["Time_Sec"] >= start_sec) & (df["Time_Sec"] < end_sec)]
                if col_idx == len(time_windows) - 1:
                    window_df = df[(df["Time_Sec"] >= start_sec) & (df["Time_Sec"] <= end_sec)]
            for row_idx, shrimp_id in enumerate(shrimp_ids):
                group = window_df[window_df["Shrimp_ID"] == shrimp_id] if not window_df.empty else pd.DataFrame()
                seen = int(len(group))
                if seen == 0:
                    continue
                male_hits = int((group["Pred_Label"] == "Male").sum())
                male_rate = male_hits / seen
                if seen < MIN_OBSERVATIONS_PER_SHRIMP:
                    matrix[row_idx, col_idx] = 3
                    labels[row_idx][col_idx] = f"U\nn={seen}"
                elif male_rate >= MALE_RATE_THRESHOLD:
                    matrix[row_idx, col_idx] = 1
                    labels[row_idx][col_idx] = f"M\n{male_hits}/{seen}"
                else:
                    matrix[row_idx, col_idx] = 2
                    labels[row_idx][col_idx] = f"F\n{male_hits}/{seen}"

        cmap = mcolors.ListedColormap(["#FFFFFF", "#2F6FDB", "#D94F70", "#8A8F98"])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        x_labels = [f"{int(row['Start_Sec'])}-{int(round(row['End_Sec']))}s" for _, row in time_windows.iterrows()]
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(shrimp_ids)))
        ax.set_yticklabels([f"ID{sid}" for sid in shrimp_ids])
        ax.set_title("各 ID 時間窗公母變化", fontsize=14, fontweight="bold")
        ax.set_xlabel("影片時間窗")
        ax.set_ylabel("Shrimp_ID")

        for row_idx in range(len(shrimp_ids)):
            for col_idx in range(len(time_windows)):
                text = labels[row_idx][col_idx]
                color = "white" if matrix[row_idx, col_idx] in (1, 2, 3) else "#444444"
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color=color)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#2F6FDB", label="Male"),
            plt.Rectangle((0, 0), 1, 1, color="#D94F70", label="Female"),
            plt.Rectangle((0, 0), 1, 1, color="#8A8F98", label=f"Unknown (<{MIN_OBSERVATIONS_PER_SHRIMP} obs)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", edgecolor="#CCCCCC", label="No observation"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_xticks(np.arange(-0.5, len(time_windows), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(shrimp_ids), 1), minor=True)
        ax.grid(which="minor", color="#D0D0D0", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def _plot_auto_total(self, counts: pd.DataFrame, summary: dict) -> str:
        path = os.path.join(self.figure_dir, "auto_total_counts.png")
        fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white", constrained_layout=True)
        if not counts.empty:
            count_values = counts["OBB_Detection_Count"].astype(int)
            recommended = int(summary["Recommended_Total_Shrimp"])
            dist = count_values.value_counts().sort_index()
            colors = ["#C23B3B" if value == recommended else "#8A8F98" for value in dist.index]
            ax.bar(dist.index.astype(str), dist.values, color=colors)
            for i, (count, frames) in enumerate(zip(dist.index, dist.values)):
                pct = frames / max(len(count_values), 1) * 100
                ax.text(i, frames + max(dist.values) * 0.02, f"n={frames}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, max(dist.values) * 1.18)
        ax.set_title("自動估計蝦子總數：OBB 偵測數分布", fontsize=14, fontweight="bold")
        ax.set_xlabel("每個抽樣幀偵測到的蝦子數")
        ax.set_ylabel("抽樣幀數")
        ax.grid(axis="y", alpha=0.25)
        text = (
            f"P{summary['Percentile_Used']:.0f}={summary['Percentile_Count']} | "
            f"Recommended={summary['Recommended_Total_Shrimp']} | "
            f"Median={summary['Median_Count']} | Max={summary['Max_Count']} | "
            f"Confidence={summary['Auto_Total_Confidence']}"
        )
        ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"})
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def _plot_single_vs_multi(self, evaluation: dict) -> str:
        path = os.path.join(self.figure_dir, "single_vs_multiframe_accuracy.png")
        labels = ["單幀判斷", "多幀彙整"]
        values = [
            float(evaluation.get("Single_Frame_Accuracy_Pct", 0) or 0),
            float(evaluation.get("Multi_Frame_Accuracy_Pct", 0) or 0),
        ]
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor="white")
        ax.bar(labels, values, color=["#8A8F98", "#2E7D32"])
        ax.set_ylim(0, 100)
        ax.set_title("單幀與多幀辨識成效比較", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        for i, value in enumerate(values):
            ax.text(i, value + 1, f"{value:.1f}%", ha="center")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def _confusion_df(self, per_shrimp: pd.DataFrame) -> pd.DataFrame:
        if "True_Label" not in per_shrimp.columns:
            return pd.DataFrame()
        valid = per_shrimp[per_shrimp["True_Label"] != ""]
        if valid.empty:
            return pd.DataFrame()
        labels = ["Male", "Female", "Unknown"]
        cm = pd.crosstab(valid["True_Label"], valid["Final_Label"], rownames=["True"], colnames=["Pred"])
        return cm.reindex(index=["Male", "Female"], columns=labels, fill_value=0)

    def _write_confusion_matrix(self, per_shrimp: pd.DataFrame) -> str:
        path = os.path.join(self.data_dir, "confusion_matrix.csv")
        self._confusion_df(per_shrimp).to_csv(path, encoding="utf-8-sig")
        return path

    def _plot_confusion_matrix(self, per_shrimp: pd.DataFrame) -> str:
        path = os.path.join(self.figure_dir, "confusion_matrix.png")
        cm = self._confusion_df(per_shrimp)
        fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="white")
        if not cm.empty:
            im = ax.imshow(cm.values, cmap="Blues")
            ax.set_xticks(range(len(cm.columns)))
            ax.set_xticklabels(cm.columns)
            ax.set_yticks(range(len(cm.index)))
            ax.set_yticklabels(cm.index)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm.values[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("公母辨識混淆矩陣", fontsize=14, fontweight="bold")
        ax.set_xlabel("系統判定")
        ax.set_ylabel("人工標註")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path
