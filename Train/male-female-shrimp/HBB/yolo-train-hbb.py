import ultralytics
from ultralytics import YOLO
import os

# 1. 確保環境
ultralytics.checks()

# 設定路徑
DATA_YAML = 'data.yaml'
MODEL_PATH = '../../yolo11m-hbb.pt' 

if __name__ == '__main__':
    # 2. 載入模型
    model = YOLO(MODEL_PATH)
    
    print("🚀 開始執行【蝦子定位裁切專用】單一類別訓練...")

    # 3. 執行訓練
    results = model.train(
        data=DATA_YAML,
        device=0,
        epochs=200,
        imgsz=416,         
        batch=16,           
        name='traindata20260520_公母蝦_hbb_1',
        exist_ok=True,
        #patience=30,
        hsv_h=0.015,   # 色調 (微調)
        hsv_s=0.7,     # 飽和度 (增加飽和度變動)
        hsv_v=0.4,     # 亮度 (Value/Brightness)，這最接近對比度調整的效果
        degrees=180.0, # 隨機旋轉，蝦子在水箱會轉任何角度
        scale=0.5,     # 縮放
        fliplr=0.5,    # 左右翻轉
        flipud=0.5,    # 上下翻轉 (仰拍必備)
        mosaic=1.0,    # 拼貼
        mixup=0.1,     # 混合
        copy_paste=0.2, # 複製貼上
    )

    print(f"✅ 訓練完成!")
