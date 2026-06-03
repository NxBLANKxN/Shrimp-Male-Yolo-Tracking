# Grass Shrimp Sex Ratio Analyzer

## 中文版

### 系統目的

本系統用於分析養殖人員拍攝的草蝦仰拍影片。使用情境是：養殖人員將一批草蝦放入透明桶中，攝影機架設於桶底下方，系統在不取出蝦子的情況下，估計桶內公蝦與母蝦數量，並輸出可人工確認的關鍵幀與相關數據。

系統重點不是長時間精準追蹤每一隻蝦，而是透過固定幀抽樣與多幀彙整，建立每隻蝦的公母辨識率，並提供影像佐證。

### 核心流程

```text
影片輸入
-> 固定幀抽樣
-> OBB 模型偵測蝦體全身
-> 將 OBB 蝦體裁切並拉正
-> HBB 模型偵測公蝦性徵 male_line
-> 固定 ID 池分配 Shrimp_ID
-> 每隻蝦累積多幀辨識結果
-> 輸出公母比例、每隻蝦辨識率、關鍵幀與評估數據
```

### 公蝦判定方式

單一幀不直接決定一隻蝦的性別。系統會累積同一個 `Shrimp_ID` 的多次觀測：

```text
Male_Rate = Male_Hits / Total_Seen
```

若：

```text
Total_Seen >= MIN_OBSERVATIONS_PER_SHRIMP
且 Male_Rate >= MALE_RATE_THRESHOLD
```

則判定為公蝦。否則若觀測數足夠但命中率低於門檻，判定為母蝦。觀測數不足則為 `Unknown`。

目前主要參數位於 `modules/config.py`：

```python
HBB_CONF = 0.60
MALE_RATE_THRESHOLD = 0.50
MIN_OBSERVATIONS_PER_SHRIMP = 3
```

### ID 指派方式

系統使用固定 ID 池，例如：

```text
--total-shrimp 6
```

則只允許：

```text
ID1 ~ ID6
```

ID 是依據蝦體中心點距離進行分配，不使用 YOLO tracking。若同一幀偵測數量超過總蝦數，多出的框會被標記為：

```text
ID_Status = overflow
Include_In_Stats = False
```

這些 overflow 框會保留在 `detections.csv`，但不納入每隻蝦統計。

### 執行方式

### 套件需求

建議使用 Python 3.9 以上版本，並在專案資料夾建立虛擬環境後安裝套件：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics pandas tqdm
```

主要套件用途：

```text
ultralytics     載入與執行 YOLO OBB/HBB 模型
opencv-python   讀取影片、影像裁切、繪製 keyframe
numpy           影像陣列與座標計算
pandas          輸出與讀取 CSV 統計資料
matplotlib      產生統計圖表
tqdm            顯示影片分析進度
```

若要使用 GPU/CUDA，請先依照 PyTorch 官方建議安裝對應版本的 `torch`，再安裝上述套件；一般 CPU 執行可直接使用上述指令。

基本執行：

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --keyframes 12 --window-sec 10
```

若已知真實公母數，可加入：

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --keyframes 12 --window-sec 10 --gt-male 3 --gt-female 3
```

若已填寫每隻蝦的人工真值，可用：

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --keyframes 12 --window-sec 10 --truth-csv "outputs\<video>\analysis_<time>\data\truth_template.csv"
```

若不確定桶內蝦子總數，可先讓系統用 OBB-only 預掃描自動估計：

```powershell
python Fast-Stats.py --video "video\未知桶.mp4" --auto-total --skip-frames 10 --keyframes 12 --window-sec 10
```

預設使用每個抽樣幀 OBB 偵測數的第 90 百分位數作為建議總數。可調整：

```powershell
python Fast-Stats.py --video "video\未知桶.mp4" --auto-total --auto-total-percentile 95 --skip-frames 10
```

也可以讓自動估數使用和正式分析不同的抽樣幀數設定：

```powershell
python Fast-Stats.py --video "video\未知桶.mp4" --auto-total --auto-total-skip-frames 5 --auto-total-max-frames 120 --skip-frames 10
```

這代表自動估數階段每 5 幀掃一次，最多使用 120 個抽樣幀；正式分析仍每 10 幀分析一次。

若要在分析時即時查看標註畫面，可加入：

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --preview
```

預覽視窗會顯示每個抽樣幀的 OBB、Shrimp_ID、M/F 與 male_line 黃色框，右側會同步顯示每一隻蝦經 OBB 裁切拉正後的 crop。crop 會統一成長軸水平，右側縮圖依 Shrimp_ID 固定排序，並會用同一 ID 的上一幀 crop 穩定左右方向，減少預覽時左右翻轉。crop 內會標出 male_line 黃框。預覽頻率由 `--skip-frames` 決定，例如 `--skip-frames 10` 代表每 10 幀顯示一次；若要接近逐幀查看，可設為 `--skip-frames 1`。按 `q` 或 `Esc` 可提前停止分析；輸出的統計會以已處理的幀為準。若畫面太大，可用 `--preview-scale` 調整：

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --preview --preview-scale 0.5
```

若只想查看即時畫面與右側 crop，不想輸出 CSV、圖表或 keyframe，可使用：

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 1 --preview-only
```

`--preview-only` 適合檢查模型與畫面，不會建立 `outputs/<video>/analysis_<timestamp>/`。

彙整多次分析結果與模型比較：

```powershell
python compare_runs.py --outputs outputs
```

只比較每部影片最新一次分析：

```powershell
python compare_runs.py --outputs outputs --latest-only
```

若需要額外輸出模型比較圖與合併混淆矩陣圖，可加：

```powershell
python compare_runs.py --outputs outputs --latest-only --extra-plots
```

### 主要參數

```text
--video          輸入影片路徑
--total-shrimp   桶中預期蝦子總數
--auto-total     使用 OBB-only 預掃描自動估計桶中蝦子總數
--auto-total-percentile 自動估總數時使用的偵測數百分位數，預設 90
--auto-total-skip-frames 自動估總數預掃描每隔幾幀偵測一次，預設同 --skip-frames
--auto-total-max-frames 自動估總數最多使用幾個抽樣幀
--skip-frames    每隔幾幀分析一次
--keyframes      輸出幾張關鍵幀圖片
--window-sec     每幾秒統計一次時間窗數據，預設 10 秒
--gt-male        選填，真實公蝦數
--gt-female      選填，真實母蝦數
--truth-csv      選填，每隻 ID 的人工真值 CSV
--preview        分析時顯示即時標註預覽視窗
--preview-only   只顯示即時預覽，不輸出 CSV、圖表或 keyframe
--preview-scale  預覽視窗縮放比例，預設 0.75
--preview-wait-ms 每個預覽幀停留毫秒數，預設 1
```

### 輸出結構

每次分析會建立：

```text
outputs/<video_name>/analysis_<timestamp>/
  data/
    bucket_summary.csv
    per_shrimp_summary.csv
    detections.csv
    auto_total_counts.csv
    auto_total_summary.csv
    evaluation_summary.csv
    reliability_summary.csv
    time_window_summary.csv
    error_cases.csv
    truth_template.csv
    keyframe_index.csv
  figures/
    sex_ratio_summary.png
    auto_total_counts.png
    per_shrimp_male_rate.png
    temporal_sex_ratio.png
    single_vs_multiframe_accuracy.png
    confusion_matrix.png
  evidence/
    keyframes/
      keyframe_*.jpg

outputs/comparison/
  model_comparison_summary.csv
  per_video_performance.png
  combined_confusion_matrix.csv
```

`single_vs_multiframe_accuracy.png`、`confusion_matrix.png` 與 `confusion_matrix.csv` 需要提供 `--truth-csv` 後才會產生。

### 重要輸出說明

#### `bucket_summary.csv`

整桶層級的分析摘要：

```text
Pred_Male
Pred_Female
Unknown
Male_Ratio_Pct
Female_Ratio_Pct
Count_Accuracy_Pct
Exact_Count_Match
```

#### `auto_total_summary.csv`

使用 `--auto-total` 時產生的自動估總數摘要：

```text
Frames_Used
Skip_Frames
Max_Frames
Percentile_Used
Recommended_Total_Shrimp
Percentile_Count
Median_Count
Max_Count
P50_Count
P95_Count
Stability_Gap_P95_P50
Auto_Total_Confidence
```

#### `auto_total_counts.csv`

使用 `--auto-total` 時，每個抽樣幀的 OBB 偵測數：

```text
Frame
Time_Sec
OBB_Detection_Count
Mean_OBB_Conf
```

#### `per_shrimp_summary.csv`

每隻蝦的多幀辨識結果：

```text
Shrimp_ID
Total_Seen
Male_Hits
Female_Hits
Male_Rate_Pct
Mean_Male_Conf
Forced_ID_Count
Forced_ID_Rate_Pct
Decision_Margin_Pct
Final_Label
Enough_Evidence
```

#### `detections.csv`

每一個抽樣幀中的偵測紀錄：

```text
Frame
Time_Sec
Shrimp_ID
Pred_Label
Male_Conf
ID_Status
ID_Distance
Include_In_Stats
Box_X1, Box_Y1, Box_X2, Box_Y2
```

#### `time_window_summary.csv`

依影片幀數與 FPS，每 10 秒統計一次：

```text
Start_Sec
End_Sec
Start_Frame
End_Frame
Observed_IDs
Male
Female
Male_Ratio_Pct
Detections
Forced_ID_Rate_Pct
```

#### `reliability_summary.csv`

不需要人工真值即可產生的系統可靠性摘要：

```text
ID_Coverage_Rate_Pct
Enough_Evidence_Rate_Pct
Overflow_Rate_Pct
Overflow_Frame_Rate_Pct
Forced_Detection_Rate_Pct
Mean_ID_Distance
Male_Line_Detection_Rate_Pct
Time_Window_Male_Ratio_Std
Keyframe_Count
```

#### `truth_template.csv`

系統會自動產生此檔案，供人工填寫每個 `Shrimp_ID` 的真實性別：

```text
Shrimp_ID
Final_Label
Total_Seen
Male_Rate_Pct
True_Label
Notes
```

填好 `True_Label` 後，再用 `--truth-csv` 重新分析，即可產生單幀與多幀準確率、混淆矩陣與錯誤案例。

### 圖表說明

```text
sex_ratio_summary.png
  整桶公母數量估計

auto_total_counts.png
  使用 OBB-only 預掃描估計蝦子總數的時間序列

per_shrimp_male_rate.png
  每隻蝦的 male_line 命中率

temporal_sex_ratio.png
  每個 Shrimp_ID 在各時間窗中的 Male/Female/Unknown 變化

single_vs_multiframe_accuracy.png
  單幀判斷與多幀彙整準確率比較

confusion_matrix.png
  公母辨識混淆矩陣，包含 Male、Female、Unknown 系統輸出
```

### 多次 Run 彙整輸出

`compare_runs.py` 會掃描 `outputs/<video_name>/analysis_<timestamp>/data/`，整合每次分析的 `bucket_summary.csv`、`per_shrimp_summary.csv`、`reliability_summary.csv` 與可用的真值評估資料。

主要輸出：

```text
model_comparison_summary.csv
  每個 run 的整桶結果、模型路徑、門檻、Unknown rate、count accuracy、macro F1、balanced accuracy 與可靠性指標

per_video_performance.png
  每部影片或每個 run 的 Pred Male / Pred Female / Unknown 堆疊圖

combined_confusion_matrix.csv
  合併所有有 True_Label 的 run，產生 Male/Female/Unknown 端到端混淆矩陣表格
```

### Keyframe 佐證圖片

關鍵幀會輸出到：

```text
evidence/keyframes/
```

圖片上會顯示：

```text
ID1 M 0.82
ID2 F
OVF
```

其中：

```text
M      系統判為公蝦觀測
F      系統判為母蝦觀測
0.82   male_line 信心度
OVF    overflow，不納入統計
黃色框 male_line 偵測位置
```

### 目前限制

- `Shrimp_ID` 是系統分配的固定 ID，不等於保證完全正確的真實個體追蹤。
- 水中遮擋、重疊、快速移動仍可能造成 ID switch。
- 若 `Forced_ID_Rate_Pct` 偏高，代表該 ID 統計可信度較低。
- 最終性別判定依賴多幀結果，不建議只看單一 keyframe 做結論。

---

## English Version

### Purpose

This system analyzes bottom-view videos of grass shrimp placed in a transparent bucket. The goal is to estimate the number and ratio of male and female shrimp without removing them from the bucket, while also exporting evidence frames for human verification.

The system is not designed as a strict long-term tracker. Instead, it uses fixed-frame sampling, a fixed ID pool, and multi-frame evidence aggregation to estimate the sex of each shrimp.

### Pipeline

```text
Input video
-> Fixed-frame sampling
-> OBB model detects whole shrimp bodies
-> OBB crops are straightened
-> HBB model detects male feature: male_line
-> Fixed ID pool assigns Shrimp_ID
-> Multi-frame observations are aggregated per ID
-> Sex ratio, per-shrimp recognition rate, figures, CSV files, and keyframes are exported
```

### Male Classification Rule

A single frame does not determine the final sex. For each `Shrimp_ID`, the system aggregates multiple observations:

```text
Male_Rate = Male_Hits / Total_Seen
```

If:

```text
Total_Seen >= MIN_OBSERVATIONS_PER_SHRIMP
and Male_Rate >= MALE_RATE_THRESHOLD
```

the shrimp is classified as male. If enough observations exist but the male rate is below the threshold, it is classified as female. If there are not enough observations, the final label is `Unknown`.

Current parameters are defined in `modules/config.py`:

```python
HBB_CONF = 0.60
MALE_RATE_THRESHOLD = 0.50
MIN_OBSERVATIONS_PER_SHRIMP = 3
```

### ID Assignment

The system uses a fixed ID pool. For example:

```text
--total-shrimp 6
```

limits IDs to:

```text
ID1 ~ ID6
```

IDs are assigned by centroid distance matching. YOLO tracking is not used. If a frame contains more detections than the expected shrimp count, extra detections are marked as:

```text
ID_Status = overflow
Include_In_Stats = False
```

Overflow detections are kept in `detections.csv` for review but are excluded from per-shrimp statistics.

### Usage

### Requirements

Python 3.9 or later is recommended. Create a virtual environment in the project folder and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics pandas tqdm
```

Package usage:

```text
ultralytics     Load and run YOLO OBB/HBB models
opencv-python   Read videos, crop images, and draw keyframes
numpy           Image arrays and coordinate calculations
pandas          Read and write CSV statistics
matplotlib      Generate summary figures
tqdm            Show analysis progress
```

For GPU/CUDA execution, install the matching `torch` build recommended by the official PyTorch instructions before installing the packages above. CPU execution can use the command above directly.

Basic run:

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --keyframes 12 --window-sec 10
```

With known bucket-level ground truth:

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --keyframes 12 --window-sec 10 --gt-male 3 --gt-female 3
```

With per-shrimp manual labels:

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --keyframes 12 --window-sec 10 --truth-csv "outputs\<video>\analysis_<time>\data\truth_template.csv"
```

If the total shrimp count is unknown, use an OBB-only prescan to estimate it:

```powershell
python Fast-Stats.py --video "video\unknown_bucket.mp4" --auto-total --skip-frames 10 --keyframes 12 --window-sec 10
```

By default, the recommended count is computed from the 90th percentile of sampled-frame OBB counts. You can adjust it:

```powershell
python Fast-Stats.py --video "video\unknown_bucket.mp4" --auto-total --auto-total-percentile 95 --skip-frames 10
```

The auto-total prescan can use a different frame sampling setting from the formal analysis:

```powershell
python Fast-Stats.py --video "video\unknown_bucket.mp4" --auto-total --auto-total-skip-frames 5 --auto-total-max-frames 120 --skip-frames 10
```

This scans every 5 frames for auto-total estimation with at most 120 sampled frames, while the formal analysis still uses every 10 frames.

To inspect annotated frames during analysis, add:

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --preview
```

The preview window shows each sampled frame with OBB, Shrimp_ID, M/F labels, and yellow male_line boxes. The right panel also shows the straightened OBB crop for each detected shrimp. Crops are normalized to a horizontal long axis, right-panel thumbnails are sorted by Shrimp_ID, and each ID is compared with its previous crop to reduce left-right flipping in the preview. male_line is drawn inside each crop when detected. Preview frequency is controlled by `--skip-frames`; for example, `--skip-frames 10` displays every 10th frame, while `--skip-frames 1` is close to frame-by-frame viewing. Press `q` or `Esc` to stop early; exported summaries will use the frames processed so far. Use `--preview-scale` if the window is too large:

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 10 --preview --preview-scale 0.5
```

To only inspect the live view and right-side crops without writing CSV files, figures, or keyframes:

```powershell
python Fast-Stats.py --video "video\公母蝦仰拍-1.mp4" --total-shrimp 6 --skip-frames 1 --preview-only
```

`--preview-only` is intended for model and image inspection. It does not create `outputs/<video>/analysis_<timestamp>/`.

Combine multiple analysis runs and compare model/run results:

```powershell
python compare_runs.py --outputs outputs
```

Compare only the latest analysis run for each video:

```powershell
python compare_runs.py --outputs outputs --latest-only
```

To also export the model comparison plot and combined confusion matrix plot:

```powershell
python compare_runs.py --outputs outputs --latest-only --extra-plots
```

### Main Arguments

```text
--video          Input video path
--total-shrimp   Expected shrimp count in the bucket
--auto-total     Estimate total shrimp count with an OBB-only prescan
--auto-total-percentile Percentile of OBB counts used by auto-total, default 90
--auto-total-skip-frames Frame interval used by the auto-total prescan, default is --skip-frames
--auto-total-max-frames Maximum sampled frames used by the auto-total prescan
--skip-frames    Analyze every N frames
--keyframes      Number of evidence keyframes to export
--window-sec     Temporal window size in seconds, default 10
--gt-male        Optional bucket-level male count
--gt-female      Optional bucket-level female count
--truth-csv      Optional per-shrimp ground-truth CSV
--preview        Show a live annotated preview window during analysis
--preview-only   Only show live preview; do not write CSV files, figures, or keyframes
--preview-scale  Preview scale factor, default 0.75
--preview-wait-ms Delay in milliseconds for each preview frame, default 1
```

### Output Structure

Each run creates:

```text
outputs/<video_name>/analysis_<timestamp>/
  data/
    bucket_summary.csv
    per_shrimp_summary.csv
    detections.csv
    auto_total_counts.csv
    auto_total_summary.csv
    evaluation_summary.csv
    reliability_summary.csv
    time_window_summary.csv
    error_cases.csv
    truth_template.csv
    keyframe_index.csv
  figures/
    sex_ratio_summary.png
    auto_total_counts.png
    per_shrimp_male_rate.png
    temporal_sex_ratio.png
    single_vs_multiframe_accuracy.png
    confusion_matrix.png
  evidence/
    keyframes/
      keyframe_*.jpg

outputs/comparison/
  model_comparison_summary.csv
  per_video_performance.png
  combined_confusion_matrix.csv
```

`single_vs_multiframe_accuracy.png`, `confusion_matrix.png`, and `confusion_matrix.csv` are generated only when `--truth-csv` is provided.

### Key CSV Files

#### `bucket_summary.csv`

Bucket-level summary:

```text
Pred_Male
Pred_Female
Unknown
Male_Ratio_Pct
Female_Ratio_Pct
Count_Accuracy_Pct
Exact_Count_Match
```

#### `auto_total_summary.csv`

Auto-total summary generated when `--auto-total` is used:

```text
Frames_Used
Skip_Frames
Max_Frames
Percentile_Used
Recommended_Total_Shrimp
Percentile_Count
Median_Count
Max_Count
P50_Count
P95_Count
Stability_Gap_P95_P50
Auto_Total_Confidence
```

#### `auto_total_counts.csv`

OBB detection count for each sampled frame when `--auto-total` is used:

```text
Frame
Time_Sec
OBB_Detection_Count
Mean_OBB_Conf
```

#### `per_shrimp_summary.csv`

Per-shrimp multi-frame recognition summary:

```text
Shrimp_ID
Total_Seen
Male_Hits
Female_Hits
Male_Rate_Pct
Mean_Male_Conf
Forced_ID_Count
Forced_ID_Rate_Pct
Decision_Margin_Pct
Final_Label
Enough_Evidence
```

#### `detections.csv`

Detection-level records:

```text
Frame
Time_Sec
Shrimp_ID
Pred_Label
Male_Conf
ID_Status
ID_Distance
Include_In_Stats
Box_X1, Box_Y1, Box_X2, Box_Y2
```

#### `time_window_summary.csv`

Statistics are computed every 10 seconds based on video FPS and frame count:

```text
Start_Sec
End_Sec
Start_Frame
End_Frame
Observed_IDs
Male
Female
Male_Ratio_Pct
Detections
Forced_ID_Rate_Pct
```

#### `reliability_summary.csv`

System reliability summary generated without manual ground truth:

```text
ID_Coverage_Rate_Pct
Enough_Evidence_Rate_Pct
Overflow_Rate_Pct
Overflow_Frame_Rate_Pct
Forced_Detection_Rate_Pct
Mean_ID_Distance
Male_Line_Detection_Rate_Pct
Time_Window_Male_Ratio_Std
Keyframe_Count
```

#### `truth_template.csv`

Template for manual per-shrimp labeling:

```text
Shrimp_ID
Final_Label
Total_Seen
Male_Rate_Pct
True_Label
Notes
```

After filling `True_Label`, rerun with `--truth-csv` to generate single-frame vs multi-frame accuracy, confusion matrix, and error cases.

### Figures

```text
sex_ratio_summary.png
  Bucket-level male/female count estimate

auto_total_counts.png
  Time series of OBB-only sampled counts used for total shrimp estimation

per_shrimp_male_rate.png
  Male feature hit rate per shrimp ID

temporal_sex_ratio.png
  Male/Female/Unknown changes for each Shrimp_ID across temporal windows

single_vs_multiframe_accuracy.png
  Single-frame vs multi-frame accuracy

confusion_matrix.png
  Male/female confusion matrix including Male, Female, and Unknown system outputs
```

### Multi-Run Comparison Outputs

`compare_runs.py` scans `outputs/<video_name>/analysis_<timestamp>/data/` and combines `bucket_summary.csv`, `per_shrimp_summary.csv`, `reliability_summary.csv`, and available ground-truth evaluation data.

Main outputs:

```text
model_comparison_summary.csv
  Per-run bucket results, model paths, thresholds, Unknown rate, count accuracy, macro F1, balanced accuracy, and reliability metrics

per_video_performance.png
  Stacked Pred Male / Pred Female / Unknown counts for each video or run

combined_confusion_matrix.csv
  Combined end-to-end Male/Female/Unknown confusion matrix table for runs with True_Label
```

### Evidence Keyframes

Keyframes are exported to:

```text
evidence/keyframes/
```

Labels shown on images:

```text
ID1 M 0.82
ID2 F
OVF
```

Meaning:

```text
M      male observation
F      female observation
0.82   male_line confidence
OVF    overflow, excluded from statistics
yellow box = projected male_line detection
```

### Current Limitations

- `Shrimp_ID` is a fixed-pool system-assigned review ID, not a guaranteed true biological identity.
- Occlusion, overlap, and fast movement may still cause ID switches.
- High `Forced_ID_Rate_Pct` indicates lower ID reliability.
- Final sex classification should be interpreted from multi-frame statistics, not a single keyframe.
