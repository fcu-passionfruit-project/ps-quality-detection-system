# 多模型測試程式，整合 Stage 1 YOLO 偵測、Stage 2_1 顏色分類與 Stage 2_2 皺褶分類，並使用 OpenCV 進行影像前處理與遮罩套用。
# 多執行緒版本，讓 Stage 2_1 與 Stage 2_2 模型能同時運算，提升整體處理效率。
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading  # 引入多執行緒模組

# ==========================================
# 1. 初始化模型與設定
# ==========================================
MODEL_STAGE1_PATH = "stage1.pt"
MODEL_STAGE2_1_PATH = "stage2_1.pt"
MODEL_STAGE2_2_PATH = "stage2_2.pt"

# 載入第一階段偵測模型、第二階段顏色分類模型與皺褶分類模型
model_stage1 = YOLO(MODEL_STAGE1_PATH)
model_stage2_1 = YOLO(MODEL_STAGE2_1_PATH)
model_stage2_2 = YOLO(MODEL_STAGE2_2_PATH)

# ==========================================
# 2. 核心影像處理函式 -> mask遮罩 & 灰階轉換 + CLAHE 增強
# ==========================================
def apply_circular_mask(crop_img, mask_scale=1.01):
    """
    將裁切後的圖片套用圓形遮罩。
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

# ==========================================
# 3. GUI 功能邏輯
# ==========================================
def open_and_process():
    # 彈出檔案選擇視窗
    file_path = filedialog.askopenfilename(
        title="請選擇一張百香果照片",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    
    if not file_path:
        return

    # 讀取原始圖片 (OpenCV BGR 格式)
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        return
    
    h_orig, w_orig = img_bgr.shape[:2]
    
    # 設定預覽圖的上限尺寸
    max_size = (320, 320)

    # ------------------------------------------
    # 框 1：處理初始照片顯示
    # ------------------------------------------
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_orig_pil = Image.fromarray(img_rgb)
    img_orig_pil.thumbnail(max_size)
    tk_img_orig = ImageTk.PhotoImage(img_orig_pil)
    
    lbl_orig_img.config(image=tk_img_orig, text="")
    lbl_orig_img.image = tk_img_orig

    # ------------------------------------------
    # 執行 Stage 1 YOLO 物件偵測
    # ------------------------------------------
    results_stage1 = model_stage1(file_path)[0]

    # ------------------------------------------
    # 框 2：處理 YOLO 偵測後的畫面顯示
    # ------------------------------------------
    im_array = results_stage1.plot()
    im_det_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    img_det_pil = Image.fromarray(im_det_rgb)
    img_det_pil.thumbnail(max_size)
    tk_img_det = ImageTk.PhotoImage(img_det_pil)
    
    lbl_yolo_img.config(image=tk_img_det, text="")
    lbl_yolo_img.image = tk_img_det

    # ------------------------------------------
    # 框 3 ~ 6：影像前處理與多模型分類邏輯
    # ------------------------------------------
    # 判斷是否偵測到任何百香果目標
    if len(results_stage1.boxes) > 0:
        box = results_stage1.boxes[0]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        
        # 進行影像邊界安全檢查
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        # 進行 ROI 裁切
        crop_img = img_bgr[y1:y2, x1:x2]
        
        if crop_img.size > 0:
            # 套用圓形遮罩
            processed_img = apply_circular_mask(crop_img, mask_scale=1.01)
            proc_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            # --- 框 3：顯示 OpenCV 處理後的彩色結果 ---
            img_proc_pil = Image.fromarray(proc_rgb)
            img_proc_pil.thumbnail(max_size)
            tk_img_proc = ImageTk.PhotoImage(img_proc_pil)
            
            lbl_cv_img.config(image=tk_img_proc, text="")
            lbl_cv_img.image = tk_img_proc
            
            # --- 框 4：轉灰階並套用 CLAHE 直方圖均衡化 ---
            gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray_img)
            
            img_gray_pil = Image.fromarray(clahe_img)
            img_gray_pil.thumbnail(max_size)
            tk_img_gray = ImageTk.PhotoImage(img_gray_pil)
            
            lbl_gray_img.config(image=tk_img_gray, text="")
            lbl_gray_img.image = tk_img_gray
            
            # 轉換為 3 頻道影像以符合 YOLO 分類模型輸入結構需求（提早轉換以利多執行緒讀取）
            clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
            
            # 初始化多執行緒區域變數，用於接收模型的輸出結果
            results_stage2_1 = None
            results_stage2_2 = None
            
            # 定義 Stage 2_1 顏色模型運算工作
            def run_stage2_1():
                nonlocal results_stage2_1
                results_stage2_1 = model_stage2_1(processed_img)[0]
                
            # 定義 Stage 2_2 皺褶模型運算工作
            def run_stage2_2():
                nonlocal results_stage2_2
                results_stage2_2 = model_stage2_2(clahe_bgr)[0]
            
            # 建立並分配執行緒
            thread_color = threading.Thread(target=run_stage2_1)
            thread_wrinkle = threading.Thread(target=run_stage2_2)
            
            # 啟動雙模型平行運算
            thread_color.start()
            thread_wrinkle.start()
            
            # 等待雙模型皆運算結束後，再進行 GUI 的資料渲染
            thread_color.join()
            thread_wrinkle.join()
            
            # --- 框 5：將前處理後的彩色影像送入 Stage 2_1 顏色分類模型後的 UI 顯示 ---
            if results_stage2_1.probs is not None:
                top1_idx_1 = results_stage2_1.probs.top1
                class_name_1 = results_stage2_1.names[top1_idx_1]
                confidence_1 = results_stage2_1.probs.top1conf.item()
                lbl_title_5.config(text=f"【5】Stage 2_1 顏色: {class_name_1} ({confidence_1:.2%})", fg="#008800")
            else:
                lbl_title_5.config(text="【5】Stage 2_1 顏色: 無法讀取", fg="#cc0000")
            
            img_cls1_pil = Image.fromarray(proc_rgb)
            img_cls1_pil.thumbnail(max_size)
            tk_img_cls1 = ImageTk.PhotoImage(img_cls1_pil)
            lbl_cls1_img.config(image=tk_img_cls1, text="")
            lbl_cls1_img.image = tk_img_cls1
            
            # --- 框 6：將處理後的灰階影像送入 Stage 2_2 皺褶分類模型後的 UI 顯示 ---
            if results_stage2_2.probs is not None:
                top1_idx_2 = results_stage2_2.probs.top1
                class_name_2 = results_stage2_2.names[top1_idx_2]
                confidence_2 = results_stage2_2.probs.top1conf.item()
                lbl_title_6.config(text=f"【6】Stage 2_2 皺褶: {class_name_2} ({confidence_2:.2%})", fg="#0000cc")
            else:
                lbl_title_6.config(text="【6】Stage 2_2 皺褶: 無法讀取", fg="#cc0000")
            
            img_cls2_pil = Image.fromarray(clahe_img)
            img_cls2_pil.thumbnail(max_size)
            tk_img_cls2 = ImageTk.PhotoImage(img_cls2_pil)
            lbl_cls2_img.config(image=tk_img_cls2, text="")
            lbl_cls2_img.image = tk_img_cls2
            
        else:
            lbl_cv_img.config(image="", text="裁切區域異常")
            lbl_cv_img.image = None
            lbl_gray_img.config(image="", text="灰階轉換異常")
            lbl_gray_img.image = None
            lbl_title_5.config(text="【5】Stage 2_1 顏色: 影像異常", fg="#cc0000")
            lbl_cls1_img.config(image="", text="無有效影像")
            lbl_cls1_img.image = None
            lbl_title_6.config(text="【6】Stage 2_2 皺褶: 影像異常", fg="#cc0000")
            lbl_cls2_img.config(image="", text="無有效影像")
            lbl_cls2_img.image = None
    else:
        lbl_cv_img.config(image="", text="⚠️ 未偵測到任何百香果")
        lbl_cv_img.image = None
        lbl_gray_img.config(image="", text="無輸入影像")
        lbl_gray_img.image = None
        lbl_title_5.config(text="【5】Stage 2_1 顏色: 未偵測到目標", fg="#cc0000")
        lbl_cls1_img.config(image="", text="無輸入影像")
        lbl_cls1_img.image = None
        lbl_title_6.config(text="【6】Stage 2_2 皺褶: 未偵測到目標", fg="#cc0000")
        lbl_cls2_img.config(image="", text="無輸入影像")
        lbl_cls2_img.image = None

# ==========================================
# 4. Tkinter 介面佈局
# ==========================================
root = tk.Tk()
root.title("百香果多階段影像辨識與前處理系統")
root.geometry("1650x950")  # 調整視窗尺寸以維持版面配置
root.config(bg="#f5f5f5")

# 頂部操作按鈕
btn = tk.Button(
    root, 
    text="匯入百香果照片 📁", 
    command=open_and_process, 
    font=("微軟正黑體", 14, "bold"),
    bg="#ffdab9",
    padx=15, pady=8
)
btn.pack(pady=15)

# 網格排列容器
main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(expand=True, fill="both", padx=15, pady=5)

# 配置網格每列均勻分配寬度 (每排最多4個)
for i in range(4):
    main_frame.columnconfigure(i, weight=1)
main_frame.rowconfigure(0, weight=1)
main_frame.rowconfigure(1, weight=1)

panel_config = {"bg": "#ffffff", "highlightthickness": 1, "highlightbackground": "#cccccc"}
font_title = ("微軟正黑體", 11, "bold")

# --- 第一排（Row 0） ---

# 框 1：初始照片
frame_1 = tk.Frame(main_frame, **panel_config)
frame_1.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
tk.Label(frame_1, text="【1】原始初始照片", font=font_title, bg="#ffffff", fg="#333333").pack(pady=5)
lbl_orig_img = tk.Label(frame_1, text="尚未匯入圖片", bg="#ffffff", fg="#999999")
lbl_orig_img.pack(expand=True, fill="both", padx=5, pady=5)

# 框 2：YOLO 偵測畫面
frame_2 = tk.Frame(main_frame, **panel_config)
frame_2.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
tk.Label(frame_2, text="【2】Stage 1 偵測結果", font=font_title, bg="#ffffff", fg="#0066cc").pack(pady=5)
lbl_yolo_img = tk.Label(frame_2, text="等待模型辨識", bg="#ffffff", fg="#999999")
lbl_yolo_img.pack(expand=True, fill="both", padx=5, pady=5)

# 框 3：OpenCV 處理後的照片 (彩色)
frame_3 = tk.Frame(main_frame, **panel_config)
frame_3.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")
tk.Label(frame_3, text="【3】OpenCV 裁切與遮罩", font=font_title, bg="#ffffff", fg="#cc6600").pack(pady=5)
lbl_cv_img = tk.Label(frame_3, text="等待影像前處理", bg="#ffffff", fg="#999999")
lbl_cv_img.pack(expand=True, fill="both", padx=5, pady=5)

# 框 4：灰階後預覽 (新增的 CLAHE 強化影像顯示)
frame_gray = tk.Frame(main_frame, **panel_config)
frame_gray.grid(row=0, column=3, padx=8, pady=8, sticky="nsew")
tk.Label(frame_gray, text="【4】灰階後預覽 (CLAHE)", font=font_title, bg="#ffffff", fg="#777777").pack(pady=5)
lbl_gray_img = tk.Label(frame_gray, text="等待灰階處理", bg="#ffffff", fg="#999999")
lbl_gray_img.pack(expand=True, fill="both", padx=5, pady=5)


# --- 第二排（Row 1） ---

# 框 5：Stage 2_1 顏色分類結果
frame_4 = tk.Frame(main_frame, **panel_config)
frame_4.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
lbl_title_5 = tk.Label(frame_4, text="【5】Stage 2_1 顏色: 等待分析", font=font_title, bg="#ffffff", fg="#333333")
lbl_title_5.pack(pady=5)
lbl_cls1_img = tk.Label(frame_4, text="等待分類模型輸入", bg="#ffffff", fg="#999999")
lbl_cls1_img.pack(expand=True, fill="both", padx=5, pady=5)

# 框 6：Stage 2_2 皺褶分類結果 (改為輸入灰階影像預覽)
frame_5 = tk.Frame(main_frame, **panel_config)
frame_5.grid(row=1, column=1, padx=8, pady=8, sticky="nsew")
lbl_title_6 = tk.Label(frame_5, text="【6】Stage 2_2 皺褶: 等待分析", font=font_title, bg="#ffffff", fg="#333333")
lbl_title_6.pack(pady=5)
lbl_cls2_img = tk.Label(frame_5, text="等待皺褶模型輸入", bg="#ffffff", fg="#999999")
lbl_cls2_img.pack(expand=True, fill="both", padx=5, pady=5)

if __name__ == "__main__":
    root.mainloop()