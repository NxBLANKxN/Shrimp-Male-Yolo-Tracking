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
MALE_RATE_THRESHOLD = 0.40
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

### 主要參數

```text
--video          輸入影片路徑
--total-shrimp   桶中預期蝦子總數
--skip-frames    每隔幾幀分析一次
--keyframes      輸出幾張關鍵幀圖片
--window-sec     每幾秒統計一次時間窗數據，預設 10 秒
--gt-male        選填，真實公蝦數
--gt-female      選填，真實母蝦數
--truth-csv      選填，每隻 ID 的人工真值 CSV
```

### 輸出結構

每次分析會建立：

```text
outputs/<video_name>/analysis_<timestamp>/
  data/
    bucket_summary.csv
    per_shrimp_summary.csv
    detections.csv
    evaluation_summary.csv
    time_window_summary.csv
    error_cases.csv
    truth_template.csv
    keyframe_index.csv
  figures/
    sex_ratio_summary.png
    per_shrimp_male_rate.png
    temporal_sex_ratio.png
    single_vs_multiframe_accuracy.png
    confusion_matrix.png
  evidence/
    keyframes/
      keyframe_*.jpg
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

per_shrimp_male_rate.png
  每隻蝦的 male_line 命中率

temporal_sex_ratio.png
  每 10 秒的公母數量變化

single_vs_multiframe_accuracy.png
  單幀判斷與多幀彙整準確率比較

confusion_matrix.png
  公母辨識混淆矩陣
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
MALE_RATE_THRESHOLD = 0.40
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

### Main Arguments

```text
--video          Input video path
--total-shrimp   Expected shrimp count in the bucket
--skip-frames    Analyze every N frames
--keyframes      Number of evidence keyframes to export
--window-sec     Temporal window size in seconds, default 10
--gt-male        Optional bucket-level male count
--gt-female      Optional bucket-level female count
--truth-csv      Optional per-shrimp ground-truth CSV
```

### Output Structure

Each run creates:

```text
outputs/<video_name>/analysis_<timestamp>/
  data/
    bucket_summary.csv
    per_shrimp_summary.csv
    detections.csv
    evaluation_summary.csv
    time_window_summary.csv
    error_cases.csv
    truth_template.csv
    keyframe_index.csv
  figures/
    sex_ratio_summary.png
    per_shrimp_male_rate.png
    temporal_sex_ratio.png
    single_vs_multiframe_accuracy.png
    confusion_matrix.png
  evidence/
    keyframes/
      keyframe_*.jpg
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

per_shrimp_male_rate.png
  Male feature hit rate per shrimp ID

temporal_sex_ratio.png
  Male/female counts every 10 seconds

single_vs_multiframe_accuracy.png
  Single-frame vs multi-frame accuracy

confusion_matrix.png
  Male/female confusion matrix
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
