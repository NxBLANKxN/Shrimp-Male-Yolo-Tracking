import ultralytics
from ultralytics import YOLO
import os

# 1. 確保環境
ultralytics.checks()

# 設定路徑
DATA_YAML = 'data.yaml'
MODEL_PATH = 'yolo11m-obb.pt' 

if __name__ == '__main__':
    # 2. 載入模型
    model = YOLO(MODEL_PATH)
    
    print("🚀 開始執行【蝦子定位裁切專用】單一類別訓練...")

    # 3. 執行訓練
    results = model.train(
        data=DATA_YAML,
        device=0,
        epochs=300,        
        imgsz=960,         
        batch=8,           
        name='traindata20260602_公母蝦_obb_1',
        exist_ok=True,
        degrees=180.0,    # 隨機旋轉，這對 OBB 模型是必備的核心增強
        flipud=0.5,       # 上下翻轉
        fliplr=0.5        # 左右翻轉
    )

    print(f"✅ 訓練完成!")
