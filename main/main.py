import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# 匯入自訂的前處理與模型管理模組
from image_processing import apply_circular_mask, process_grayscale_clahe
from model_manager import PassionFruitModelManager

class PassionFruitApp:
    """
    百香果多階段影像辨識系統的主程式視窗類別。
    負責 Tkinter GUI 佈局、事件綁定與多執行緒推論協調。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("百香果多階段影像辨識與前處理系統")
        self.root.geometry("1650x950")
        self.root.config(bg="#f5f5f5")
        
        # 初始化模型管理模組，載入 Stage 1 YOLO 與 Stage 2_1、Stage 2_2 的分類模型
        self.model_manager = PassionFruitModelManager(
            stage1_path="stage1.pt",
            stage2_1_path="stage2_1.pt",
            stage2_2_path="stage2_2.pt"
        )
        
        # 建立前端介面佈局
        self.create_widgets()

    def create_widgets(self):
        """
        建構 GUI 介面元件與網格排版 (前端配置)。
        """
        # 頂部操作按鈕
        btn = tk.Button(
            self.root, 
            text="匯入百香果照片 📁", 
            command=self.open_and_process, 
            font=("微軟正黑體", 14, "bold"),
            bg="#ffdab9",
            padx=15, pady=8
        )
        btn.pack(pady=15)

        # 網格排列容器
        main_frame = tk.Frame(self.root, bg="#f5f5f5")
        main_frame.pack(expand=True, fill="both", padx=15, pady=5)

        for i in range(4):
            main_frame.columnconfigure(i, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        panel_config = {"bg": "#ffffff", "highlightthickness": 1, "highlightbackground": "#cccccc"}
        font_title = ("微軟正黑體", 11, "bold")

        # --- 第一排（Row 0） ---
        frame_1 = tk.Frame(main_frame, **panel_config)
        frame_1.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        tk.Label(frame_1, text="【1】原始初始照片", font=font_title, bg="#ffffff", fg="#333333").pack(pady=5)
        self.lbl_orig_img = tk.Label(frame_1, text="尚未匯入圖片", bg="#ffffff", fg="#999999")
        self.lbl_orig_img.pack(expand=True, fill="both", padx=5, pady=5)

        frame_2 = tk.Frame(main_frame, **panel_config)
        frame_2.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        tk.Label(frame_2, text="【2】Stage 1 偵測結果", font=font_title, bg="#ffffff", fg="#0066cc").pack(pady=5)
        self.lbl_yolo_img = tk.Label(frame_2, text="等待模型辨識", bg="#ffffff", fg="#999999")
        self.lbl_yolo_img.pack(expand=True, fill="both", padx=5, pady=5)

        frame_3 = tk.Frame(main_frame, **panel_config)
        frame_3.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")
        tk.Label(frame_3, text="【3】OpenCV 裁切與遮罩", font=font_title, bg="#ffffff", fg="#cc6600").pack(pady=5)
        self.lbl_cv_img = tk.Label(frame_3, text="等待影像前處理", bg="#ffffff", fg="#999999")
        self.lbl_cv_img.pack(expand=True, fill="both", padx=5, pady=5)

        frame_gray = tk.Frame(main_frame, **panel_config)
        frame_gray.grid(row=0, column=3, padx=8, pady=8, sticky="nsew")
        tk.Label(frame_gray, text="【4】灰階後預覽 (CLAHE)", font=font_title, bg="#ffffff", fg="#777777").pack(pady=5)
        self.lbl_gray_img = tk.Label(frame_gray, text="等待灰階處理", bg="#ffffff", fg="#999999")
        self.lbl_gray_img.pack(expand=True, fill="both", padx=5, pady=5)

        # --- 第二排（Row 1） ---
        frame_4 = tk.Frame(main_frame, **panel_config)
        frame_4.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        self.lbl_title_5 = tk.Label(frame_4, text="【5】Stage 2_1 顏色: 等待分析", font=font_title, bg="#ffffff", fg="#333333")
        self.lbl_title_5.pack(pady=5)
        self.lbl_cls1_img = tk.Label(frame_4, text="等待分類模型輸入", bg="#ffffff", fg="#999999")
        self.lbl_cls1_img.pack(expand=True, fill="both", padx=5, pady=5)

        frame_5 = tk.Frame(main_frame, **panel_config)
        frame_5.grid(row=1, column=1, padx=8, pady=8, sticky="nsew")
        self.lbl_title_6 = tk.Label(frame_5, text="【6】Stage 2_2 皺褶: 等待分析", font=font_title, bg="#ffffff", fg="#333333")
        self.lbl_title_6.pack(pady=5)
        self.lbl_cls2_img = tk.Label(frame_5, text="等待皺褶模型輸入", bg="#ffffff", fg="#999999")
        self.lbl_cls2_img.pack(expand=True, fill="both", padx=5, pady=5)

    def display_image(self, label, img_array, mode="BGR"):
        """
        核心渲染工具：將 OpenCV 影像轉換並即時更新顯示於指定的 UI Label 元件上。
        
        參數:
            label: 目標 Tkinter Label 元件。
            img_array: OpenCV 影像矩陣 (Numpy Array)。
            mode: 來源影像格式，可選 "BGR"、"RGB" 或 "GRAY"。
        """
        if mode == "BGR":
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif mode == "GRAY":
            img_rgb = img_array  # PIL 可直接識別單通道灰階
        else:
            img_rgb = img_array

        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((320, 320))
        tk_img = ImageTk.PhotoImage(pil_img)
        
        label.config(image=tk_img, text="")
        label.image = tk_img  # 保持圖片引用，防止被 Garbage Collection 機制銷毀

    def reset_ui_on_error(self, message_main, message_sub):
        """
        當未偵測到目標或影像異常時，重置影像顯示標籤與文字提示。
        """
        self.lbl_cv_img.config(image="", text=message_main)
        self.lbl_cv_img.image = None
        self.lbl_gray_img.config(image="", text=message_sub)
        self.lbl_gray_img.image = None
        self.lbl_title_5.config(text=f"【5】Stage 2_1 顏色: {message_main}", fg="#cc0000")
        self.lbl_cls1_img.config(image="", text="無有效影像")
        self.lbl_cls1_img.image = None
        self.lbl_title_6.config(text=f"【6】Stage 2_2 皺褶: {message_main}", fg="#cc0000")
        self.lbl_cls2_img.config(image="", text="無有效影像")
        self.lbl_cls2_img.image = None

    def open_and_process(self):
        """
        核心後端邏輯：包含檔案讀取、前處理工具呼叫、多執行緒平行推論調度與資料同步。
        """
        file_path = filedialog.askopenfilename(
            title="請選擇一張百香果照片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return
        
        h_orig, w_orig = img_bgr.shape[:2]

        # 1. 顯示原始初始照片
        self.display_image(self.lbl_orig_img, img_bgr, mode="BGR")

        # 2. 執行 Stage 1 YOLO 物件偵測並顯示
        results_stage1 = self.model_manager.detect_stage1(file_path)
        im_array = results_stage1.plot()
        self.display_image(self.lbl_yolo_img, im_array, mode="BGR")

        # 3. 檢查物件框並執行前處理與多模型分類
        if len(results_stage1.boxes) > 0:
            box = results_stage1.boxes[0]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            
            crop_img = img_bgr[y1:y2, x1:x2]
            
            if crop_img.size > 0:
                # 呼叫自訂前處理模組：圓形遮罩與灰階 CLAHE
                processed_img = apply_circular_mask(crop_img, mask_scale=1.01)
                self.display_image(self.lbl_cv_img, processed_img, mode="BGR")
                
                clahe_img = process_grayscale_clahe(processed_img, clip_limit=2.0, tile_size=8)
                self.display_image(self.lbl_gray_img, clahe_img, mode="GRAY")
                
                # 初始化多執行緒執行結果接收變數
                results_stage2_1 = None
                results_stage2_2 = None
                
                # 定義執行緒任務
                def run_stage2_1():
                    nonlocal results_stage2_1
                    results_stage2_1 = self.model_manager.classify_color(processed_img)
                    
                def run_stage2_2():
                    nonlocal results_stage2_2
                    results_stage2_2 = self.model_manager.classify_wrinkle(clahe_img)
                
                # 配置並啟動雙執行緒進行平行運算
                thread_color = threading.Thread(target=run_stage2_1)
                thread_wrinkle = threading.Thread(target=run_stage2_2)
                
                thread_color.start()
                thread_wrinkle.start()
                
                # 阻擋主執行緒以等待背景運算完成
                thread_color.join()
                thread_wrinkle.join()
                
                # 4. 更新 Stage 2_1 顏色分類 UI 資訊
                if results_stage2_1.probs is not None:
                    top1_idx_1 = results_stage2_1.probs.top1
                    class_name_1 = results_stage2_1.names[top1_idx_1]
                    confidence_1 = results_stage2_1.probs.top1conf.item()
                    self.lbl_title_5.config(text=f"【5】Stage 2_1 顏色: {class_name_1} ({confidence_1:.2%})", fg="#008800")
                else:
                    self.lbl_title_5.config(text="【5】Stage 2_1 顏色: 無法讀取", fg="#cc0000")
                self.display_image(self.lbl_cls1_img, processed_img, mode="BGR")
                
                # 5. 更新 Stage 2_2 皺褶分類 UI 資訊
                if results_stage2_2.probs is not None:
                    top1_idx_2 = results_stage2_2.probs.top1
                    class_name_2 = results_stage2_2.names[top1_idx_2]
                    confidence_2 = results_stage2_2.probs.top1conf.item()
                    self.lbl_title_6.config(text=f"【6】Stage 2_2 皺褶: {class_name_2} ({confidence_2:.2%})", fg="#0000cc")
                else:
                    self.lbl_title_6.config(text="【6】Stage 2_2 皺褶: 無法讀取", fg="#cc0000")
                self.display_image(self.lbl_cls2_img, clahe_img, mode="GRAY")
                
            else:
                self.reset_ui_on_error("裁切區域異常", "灰階轉換異常")
        else:
            self.reset_ui_on_error("⚠️ 未偵測到任何百香果", "無輸入影像")

if __name__ == "__main__":
    root = tk.Tk()
    app = PassionFruitApp(root)
    root.mainloop()