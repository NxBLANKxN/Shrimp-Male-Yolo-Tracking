import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = ["Male", "Female", "Unknown"]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine shrimp analysis runs into paper-ready comparison outputs.")
    parser.add_argument("--outputs", default="outputs", help="Root folder that contains per-video analysis outputs.")
    parser.add_argument("--out-dir", help="Output folder for comparison files. Default: <outputs>/comparison")
    parser.add_argument("--latest-only", action="store_true", help="Use only the latest analysis run for each video.")
    parser.add_argument("--extra-plots", action="store_true", help="Also export model comparison and combined confusion matrix plots.")
    return parser


def read_one_row(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def analysis_dirs(outputs_root: Path) -> list[Path]:
    return sorted(
        path
        for path in outputs_root.glob("*/analysis_*")
        if path.is_dir() and (path / "data" / "bucket_summary.csv").exists()
    )


def latest_per_video(paths: list[Path]) -> list[Path]:
    latest = {}
    for path in paths:
        video = path.parent.name
        if video not in latest or path.name > latest[video].name:
            latest[video] = path
    return sorted(latest.values())


def normalize_number(value, default=0.0):
    if value == "" or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def per_id_metrics(per_shrimp: pd.DataFrame) -> dict:
    if per_shrimp.empty or "True_Label" not in per_shrimp.columns:
        return {}
    valid = per_shrimp[per_shrimp["True_Label"].fillna("").astype(str).str.strip() != ""].copy()
    if valid.empty:
        return {}

    valid["True_Label"] = valid["True_Label"].astype(str).str.strip().str.capitalize()
    valid["Final_Label"] = valid["Final_Label"].astype(str).str.strip().str.capitalize()
    support = int(len(valid))
    accuracy = float((valid["True_Label"] == valid["Final_Label"]).mean())

    recalls = []
    f1s = []
    for label in ["Male", "Female"]:
        tp = int(((valid["True_Label"] == label) & (valid["Final_Label"] == label)).sum())
        fp = int(((valid["True_Label"] != label) & (valid["Final_Label"] == label)).sum())
        fn = int(((valid["True_Label"] == label) & (valid["Final_Label"] != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)

    return {
        "Per_ID_Truth_Count": support,
        "Per_ID_Accuracy_Pct": round(accuracy * 100, 2),
        "Macro_F1_Pct": round(sum(f1s) / len(f1s) * 100, 2),
        "Balanced_Accuracy_Pct": round(sum(recalls) / len(recalls) * 100, 2),
    }


def confusion_from_per_shrimp(per_shrimp: pd.DataFrame) -> pd.DataFrame:
    if per_shrimp.empty or "True_Label" not in per_shrimp.columns:
        return empty_confusion()
    valid = per_shrimp[per_shrimp["True_Label"].fillna("").astype(str).str.strip() != ""].copy()
    if valid.empty:
        return empty_confusion()
    valid["True_Label"] = valid["True_Label"].astype(str).str.strip().str.capitalize()
    valid["Final_Label"] = valid["Final_Label"].astype(str).str.strip().str.capitalize()
    cm = pd.crosstab(valid["True_Label"], valid["Final_Label"], rownames=["True"], colnames=["Pred"])
    return cm.reindex(index=["Male", "Female"], columns=LABELS, fill_value=0)


def empty_confusion() -> pd.DataFrame:
    return pd.DataFrame(0, index=pd.Index(["Male", "Female"], name="True"), columns=pd.Index(LABELS, name="Pred"))


def collect_run(path: Path) -> tuple[dict, pd.DataFrame]:
    data_dir = path / "data"
    bucket = read_one_row(data_dir / "bucket_summary.csv")
    reliability = read_one_row(data_dir / "reliability_summary.csv")
    evaluation = read_one_row(data_dir / "evaluation_summary.csv")
    per_shrimp_path = data_dir / "per_shrimp_summary.csv"
    per_shrimp = pd.read_csv(per_shrimp_path) if per_shrimp_path.exists() else pd.DataFrame()

    row = {}
    row.update(bucket)
    row.update({k: v for k, v in reliability.items() if k not in row})
    row.update({k: v for k, v in evaluation.items() if k not in row})
    row.update(per_id_metrics(per_shrimp))

    pred_male = normalize_number(row.get("Pred_Male"))
    pred_female = normalize_number(row.get("Pred_Female"))
    unknown = normalize_number(row.get("Unknown"))
    total_ids = normalize_number(row.get("Total_Shrimp"), pred_male + pred_female + unknown)
    if total_ids <= 0:
        total_ids = pred_male + pred_female + unknown

    gt_male = normalize_number(row.get("GT_Male"), None)
    gt_female = normalize_number(row.get("GT_Female"), None)
    row["Video"] = row.get("Video") or path.parent.name
    row["Run_ID"] = path.name.replace("analysis_", "")
    row["Analysis_Dir"] = str(path)
    row["Total_IDs"] = int(total_ids) if total_ids else ""
    row["Unknown_Rate_Pct"] = round(unknown / total_ids * 100, 2) if total_ids else 0.0
    row["Male_Count_Error"] = abs(pred_male - gt_male) if gt_male is not None else ""
    row["Female_Count_Error"] = abs(pred_female - gt_female) if gt_female is not None else ""
    if gt_male is not None and gt_female is not None:
        gt_total = max(gt_male + gt_female, 1)
        row["Sex_Ratio_Error_Pct"] = round(abs((pred_male / gt_total) - (gt_male / gt_total)) * 100, 2)
    else:
        row["Sex_Ratio_Error_Pct"] = ""
    row["Run_Label"] = f"{row['Video']}\n{row['Run_ID']}"
    return row, confusion_from_per_shrimp(per_shrimp)


def write_summary(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    preferred = [
        "Video",
        "Run_ID",
        "OBB_Model",
        "HBB_Model",
        "IMGSZ_OBB",
        "IMGSZ_HBB",
        "HBB_CONF",
        "MALE_RATE_THRESHOLD",
        "MIN_OBSERVATIONS_PER_SHRIMP",
        "Pred_Male",
        "Pred_Female",
        "Unknown",
        "Unknown_Rate_Pct",
        "Male_Ratio_Pct",
        "GT_Male",
        "GT_Female",
        "Count_Accuracy_Pct",
        "Exact_Count_Match",
        "Sex_Ratio_Error_Pct",
        "Per_ID_Accuracy_Pct",
        "Macro_F1_Pct",
        "Balanced_Accuracy_Pct",
        "Per_ID_Truth_Count",
        "Mean_Observations_Per_Shrimp",
        "ID_Coverage_Rate_Pct",
        "Enough_Evidence_Rate_Pct",
        "Overflow_Rate_Pct",
        "Forced_Detection_Rate_Pct",
        "Time_Window_Male_Ratio_Std",
        "Analysis_Dir",
    ]
    ordered = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]
    df = df[ordered].sort_values(["Video", "Run_ID"])
    df.to_csv(out_dir / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    return df


def plot_model_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    metric = "Count_Accuracy_Pct" if "Count_Accuracy_Pct" in plot_df.columns and plot_df["Count_Accuracy_Pct"].notna().any() else "Male_Ratio_Pct"
    values = pd.to_numeric(plot_df[metric], errors="coerce").fillna(0)
    unknown = pd.to_numeric(plot_df.get("Unknown_Rate_Pct", pd.Series([0] * len(plot_df))), errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(9, max(5.2, len(plot_df) * 0.55)), facecolor="white")
    y = range(len(plot_df))
    ax.barh(y, values, color="#2F6FDB", label=metric)
    ax.scatter(unknown, y, color="#C23B3B", marker="o", s=45, label="Unknown_Rate_Pct", zorder=3)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent")
    ax.set_title("Model / Run Comparison", fontsize=14, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["Run_Label"], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    for i, value in enumerate(values):
        ax.text(value + 1.2, i, f"{value:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_video_counts(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    labels = df["Run_Label"].tolist()
    male = pd.to_numeric(df.get("Pred_Male", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    female = pd.to_numeric(df.get("Pred_Female", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    unknown = pd.to_numeric(df.get("Unknown", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(9, max(5.2, len(df) * 0.55)), facecolor="white")
    ax.barh(y, male, color="#2F6FDB", label="Pred Male")
    ax.barh(y, female, left=male, color="#D94F70", label="Pred Female")
    ax.barh(y, unknown, left=male + female, color="#8A8F98", label="Unknown")
    ax.set_title("Per-Video Predicted Sex Counts", fontsize=14, fontweight="bold")
    ax.set_xlabel("Shrimp count")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    for i, total in enumerate(male + female + unknown):
        ax.text(total + 0.08, i, f"n={int(total)}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "per_video_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: pd.DataFrame, out_dir: Path) -> None:
    cm.to_csv(out_dir / "combined_confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.2, 4.8), facecolor="white")
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_title("Combined Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("System output")
    ax.set_ylabel("Manual label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm.values[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_dir / "combined_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    outputs_root = Path(args.outputs)
    out_dir = Path(args.out_dir) if args.out_dir else outputs_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = analysis_dirs(outputs_root)
    if args.latest_only:
        paths = latest_per_video(paths)
    if not paths:
        raise FileNotFoundError(f"No analysis runs found under {outputs_root}")

    rows = []
    combined_cm = empty_confusion()
    for path in paths:
        row, cm = collect_run(path)
        rows.append(row)
        combined_cm = combined_cm.add(cm, fill_value=0).astype(int)

    df = write_summary(rows, out_dir)
    plot_per_video_counts(df, out_dir)
    combined_cm.to_csv(out_dir / "combined_confusion_matrix.csv", encoding="utf-8-sig")
    if args.extra_plots:
        plot_model_comparison(df, out_dir)
        plot_confusion(combined_cm, out_dir)

    print(f"Runs combined: {len(df)}")
    print(f"Output: {os.path.abspath(out_dir)}")
    print(f"- {out_dir / 'model_comparison_summary.csv'}")
    print(f"- {out_dir / 'per_video_performance.png'}")
    print(f"- {out_dir / 'combined_confusion_matrix.csv'}")
    if args.extra_plots:
        print(f"- {out_dir / 'model_comparison.png'}")
        print(f"- {out_dir / 'combined_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
