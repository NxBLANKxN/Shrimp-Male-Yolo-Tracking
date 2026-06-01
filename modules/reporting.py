# modules/reporting.py

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


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
        paths["ratio_chart"] = self._plot_ratio(summary)
        paths["shrimp_chart"] = self._plot_per_shrimp(per_shrimp)
        paths["temporal_chart"] = self._plot_temporal(time_windows)
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
        ax.set_title("整桶草蝦公母數量估計", fontsize=14, fontweight="bold")
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
        ax.axhline(50, color="#222222", linestyle="--", label="公蝦判定門檻 50%")
        ax.set_ylim(0, 105)
        ax.set_title("每隻蝦的公蝦特徵命中率", fontsize=14, fontweight="bold")
        ax.set_xlabel("系統分配 ID")
        ax.set_ylabel("公蝦特徵命中率 (%)")
        ax.legend()
        for i, row in per_shrimp.iterrows():
            ax.text(i, row["Male_Rate_Pct"] + 2, f"n={row['Total_Seen']}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def _plot_temporal(self, time_windows: pd.DataFrame) -> str:
        path = os.path.join(self.figure_dir, "temporal_sex_ratio.png")
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        if not time_windows.empty:
            ax.plot(time_windows["Start_Sec"], time_windows["Male"], marker="o", label="公蝦", color="#2F6FDB")
            ax.plot(time_windows["Start_Sec"], time_windows["Female"], marker="o", label="母蝦", color="#D94F70")
            ax.plot(time_windows["Start_Sec"], time_windows["Observed_IDs"], linestyle="--", label="觀測 ID 數", color="#525866")
            starts = time_windows["Start_Sec"].tolist()
            ax.set_xticks(starts)
            ax.set_xlim(min(starts), max(time_windows["End_Sec"].tolist()))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.set_title("時間窗公母數量穩定性", fontsize=14, fontweight="bold")
        ax.set_xlabel("影片時間 (秒，每 10 秒統計一次)")
        ax.set_ylabel("數量")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()
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
        valid = per_shrimp[(per_shrimp["True_Label"] != "") & (per_shrimp["Final_Label"] != "Unknown")]
        if valid.empty:
            return pd.DataFrame()
        return pd.crosstab(valid["True_Label"], valid["Final_Label"], rownames=["True"], colnames=["Pred"])

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
