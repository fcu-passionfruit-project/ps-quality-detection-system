# ROI照片_批量處理程式
import os
import cv2
import numpy as np
from ultralytics import YOLO

def apply_circular_mask(crop_img, mask_scale=1.2):
    """
    將裁切後的圖片套用圓形遮罩。
    透過 mask_scale 調整遮罩範圍：
    - 1.0: 剛好貼齊邊緣（黑色最多）
    - 1.2: 圓形放大 20%，四個角落的黑色會變少
    - 1.5: 圓形放大 50%，只會遮到最極端的尖角
    """
    h, w = crop_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 根據放大倍率，計算新的水平與垂直半徑
    rx = int((w // 2) * mask_scale)
    ry = int((h // 2) * mask_scale)
    
    # 在中心繪製放大後的圓形/橢圓
    cv2.ellipse(mask, (w // 2, h // 2), (rx, ry), 0, 0, 360, 255, -1)
    
    # 套用遮罩
    masked_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)
    return masked_img

def batch_process_passionfruit(model_path, input_dir, output_dir, conf_threshold=0.3, shrink_ratio=0.0):
    """
    批量讀取資料夾照片，自動偵測百香果、進行內縮/遮罩處理並另存新檔。
    """
    # 載入訓練好的 stage1.pt 模型
    model = YOLO(model_path)
    
    # 自動建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    # 支援的常見圖片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f" 找到 {len(img_files)} 張圖片")
    
    count = 0
    for img_name in img_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h_orig, w_orig = img.shape[:2]
        
        # 進行 YOLO 偵測
        results = model(img, conf=conf_threshold)[0]
        
        # 遍歷每一顆偵測到的百香果
        for idx, box in enumerate(results.boxes):
            # 取得 xyxy 座標
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            # 確保座標沒有超出原始圖片邊界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            
            # 【影像處理功能 1】如果需要邊界等比例內縮 (例如 shrink_ratio=0.05 代表四周內縮 5%)
            if shrink_ratio > 0:
                box_w = x2 - x1
                box_h = y2 - y1
                pad_w = int(box_w * shrink_ratio)
                pad_h = int(box_h * shrink_ratio)
                x1 = min(x2, x1 + pad_w)
                y1 = min(y2, y1 + pad_h)
                x2 = max(x1, x2 - pad_w)
                y2 = max(y1, y2 - pad_h)
            
            # 進行 ROI 裁切
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue
            
            # 【影像處理功能 2】套用圓形遮罩，遮住四個角落
            processed_img = apply_circular_mask(crop_img, mask_scale=1.01)
            
            # 產生新的檔名（原檔名 + _crop_序號）
            base_name, ext = os.path.splitext(img_name)
            save_name = f"{base_name}_crop_{idx}{ext}"
            save_path = os.path.join(output_dir, save_name)
            
            # 儲存結果
            cv2.imwrite(save_path, processed_img)
            count += 1
            
    print(f"總共裁切並處理了 {count} 顆百香果。")

if __name__ == "__main__":
    # ==========================================
    # 請在這裡換成你的實際路徑
    # ==========================================
    MODEL_PATH = "stage1.pt"                       # 你的第一層 YOLO 模型路徑
    INPUT_FOLDER = r"D:\YOLO_Training\passionFruit\Dataset\百香果照片集\test"       # 含有幾百張原始照片的資料夾
    OUTPUT_FOLDER = r"D:\YOLO_Training\passionFruit\Dataset\百香果照片集\test_cropper"  # 處理完要存放的新資料夾
    
    # 執行批量處理
    # 如果想同時搭配內縮，可以把 shrink_ratio 改成 0.05 (四周各內縮 5%)
    batch_process_passionfruit(
        model_path=MODEL_PATH, 
        input_dir=INPUT_FOLDER, 
        output_dir=OUTPUT_FOLDER, 
        conf_threshold=0.3,    # 信心度閾值，可自由調整
        shrink_ratio=0.0       # 內縮比例（0.0 代表不內縮，直接套用圓形遮罩）
    )