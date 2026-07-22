# 帶有GUI版本的 ROI照片_批量處理程式
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from ultralytics import YOLO

# ==================== 核心影像處理邏輯 ====================

def apply_circular_mask(crop_img, mask_scale):
    """
    將裁切後的圖片套用圓形遮罩，遮住四個角落的背景。
    透過 mask_scale 調整遮罩範圍：
    - 1.0: 剛好貼齊邊緣（黑色最多）
    - 1.2: 圓形放大 20%，四個角落的黑色會變少
    - 1.5: 圓形放大 50%，只會遮到最極端的尖角
    """
    h, w = crop_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    rx = int((w // 2) * mask_scale)
    ry = int((h // 2) * mask_scale)
    
    cv2.ellipse(mask, (w // 2, h // 2), (rx, ry), 0, 0, 360, 255, -1)
    return cv2.bitwise_and(crop_img, crop_img, mask=mask)

# ==================== GUI 介面設計 ====================

class CropperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("百香果 ROI 批量裁切產生器")
        self.root.geometry("650x650")
        self.root.minsize(550, 550)
        
        # 設定路徑變數
        self.model_path = tk.StringVar(value="stage1.pt") # 預設模型路徑
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        # 建立介面元件
        self.create_widgets()
        
    def log(self, message):
        """將訊息輸出到下方的日誌文字框中"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def create_widgets(self):
        # 1. 路徑選取區塊 (Labelframe)
        path_frame = ttk.LabelFrame(self.root, text=" 檔案與路徑設定 ", padding=10)
        path_frame.pack(fill="x", padx=15, pady=10)
        
        # 模型路徑
        ttk.Label(path_frame, text="YOLO 模型 (.pt):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(path_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="瀏覽...", command=self.select_model).grid(row=0, column=2, padx=5)
        
        # 輸入資料夾
        ttk.Label(path_frame, text="原始照片資料夾:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(path_frame, textvariable=self.input_dir, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="選取...", command=self.select_input).grid(row=1, column=2, padx=5)
        
        # 輸出資料夾
        ttk.Label(path_frame, text="另存裁切資料夾:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(path_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(path_frame, text="選取...", command=self.select_output).grid(row=2, column=2, padx=5)
        
        # 2. 參數調整區塊 (滑條)
        param_frame = ttk.LabelFrame(self.root, text=" 參數動態調整 ", padding=10)
        param_frame.pack(fill="x", padx=15, pady=10)
        
        # 信心度 (conf_threshold)
        ttk.Label(param_frame, text="偵測信心度 (conf):").grid(row=0, column=0, sticky="w", pady=5)
        self.conf_slider = ttk.Scale(param_frame, from_=0.1, to=1.0, value=0.3, command=self.update_labels)
        self.conf_slider.grid(row=0, column=1, sticky="we", padx=10, pady=5)
        self.conf_label = ttk.Label(param_frame, text="0.30")
        self.conf_label.grid(row=0, column=2, padx=5)
        
        # 內縮比例 (shrink_ratio)
        ttk.Label(param_frame, text="邊界內縮比例 (shrink):").grid(row=1, column=0, sticky="w", pady=5)
        self.shrink_slider = ttk.Scale(param_frame, from_=0.0, to=0.3, value=0.0, command=self.update_labels)
        self.shrink_slider.grid(row=1, column=1, sticky="we", padx=10, pady=5)
        self.shrink_label = ttk.Label(param_frame, text="0.00")
        self.shrink_label.grid(row=1, column=2, padx=5)
        
        # 遮罩放大倍率 (mask_scale)
        ttk.Label(param_frame, text="遮罩放大倍率 (mask):").grid(row=2, column=0, sticky="w", pady=5)
        self.mask_slider = ttk.Scale(param_frame, from_=1.0, to=2.0, value=1.01, command=self.update_labels)
        self.mask_slider.grid(row=2, column=1, sticky="we", padx=10, pady=5)
        self.mask_label = ttk.Label(param_frame, text="1.01")
        self.mask_label.grid(row=2, column=2, padx=5)
        
        # 讓滑條那欄可以自動伸縮
        param_frame.columnconfigure(1, weight=1)
        
        # 3. 開始按鈕
        self.start_btn = ttk.Button(self.root, text="🚀 開始批量裁切處理", command=self.start_processing_thread)
        self.start_btn.pack(pady=10, ipadx=20, ipady=5)
        
        # 4. 狀態日誌區塊
        log_frame = ttk.LabelFrame(self.root, text=" 執行進度與日誌 ", padding=5)
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.log_text = ScrolledText(log_frame, height=12, bg="#f5f5f5", font=("Microsoft JhengHei", 9))
        self.log_text.pack(fill="both", expand=True)
        self.log("已準備就緒，請設定好路徑與參數後按下按鈕。")

    # ==================== 按鈕觸發事件 ====================
    
    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if file_path:
            self.model_path.set(file_path)
            
    def select_input(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.input_dir.set(dir_path)
            
    def select_output(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)
            
    def update_labels(self, event=None):
        """即時把滑條的數值四捨五入顯示在標籤上"""
        self.conf_label.config(text=f"{self.conf_slider.get():.2f}")
        self.shrink_label.config(text=f"{self.shrink_slider.get():.2f}")
        self.mask_label.config(text=f"{self.mask_slider.get():.2f}")

    def start_processing_thread(self):
        """使用多執行緒啟動，防止 GUI 卡死"""
        if not self.input_dir.get() or not self.output_dir.get() or not self.model_path.get():
            messagebox.showwarning("提示", "路徑還沒選填完整！")
            return
        
        # 停用按鈕，避免重複點擊
        self.start_btn.config(state="disabled")
        
        # 建立執行緒跑後台處理
        t = threading.Thread(target=self.process_core)
        t.daemon = True
        t.start()

    # ==================== 後台批次處理核心 ====================

    def process_core(self):
        try:
            model_p = self.model_path.get()
            in_dir = self.input_dir.get()
            out_dir = self.output_dir.get()
            
            # 獲取滑條當前的數值
            conf_val = float(self.conf_slider.get())
            shrink_val = float(self.shrink_slider.get())
            mask_val = float(self.mask_slider.get())
            
            self.log(f"\n[系統] 開始載入 YOLO 模型: {os.path.basename(model_p)} ...")
            model = YOLO(model_p)
            
            os.makedirs(out_dir, exist_ok=True)
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            img_files = [f for f in os.listdir(in_dir) if f.lower().endswith(valid_extensions)]
            
            self.log(f"[系統] 找到 {len(img_files)} 張圖片，開始批量處理...")
            
            count = 0
            for current_idx, img_name in enumerate(img_files, 1):
                img_path = os.path.join(in_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                h_orig, w_orig = img.shape[:2]
                results = model(img, conf=conf_val, verbose=False)[0]
                
                for idx, box in enumerate(results.boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_orig, x2), min(h_orig, y2)
                    
                    # 內縮處理
                    if shrink_val > 0:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        pad_w = int(box_w * shrink_val)
                        pad_h = int(box_h * shrink_val)
                        x1 = min(x2, x1 + pad_w)
                        y1 = min(y2, y1 + pad_h)
                        x2 = max(x1, x2 - pad_w)
                        y2 = max(y1, y2 - pad_h)
                    
                    crop_img = img[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        continue
                    
                    # 遮罩處理 (套用畫面上拉的數值)
                    processed_img = apply_circular_mask(crop_img, mask_scale=mask_val)
                    
                    base_name, ext = os.path.splitext(img_name)
                    save_name = f"{base_name}_crop_{idx}{ext}"
                    save_path = os.path.join(out_dir, save_name)
                    
                    cv2.imwrite(save_path, processed_img)
                    count += 1
                
                if current_idx % 10 == 0 or current_idx == len(img_files):
                    self.log(f"[進度] 已處理 {current_idx}/{len(img_files)} 張圖片...")
            
            self.log(f"🎉 [成功] 總共裁切出 {count} 顆百香果。")
            messagebox.showinfo("完成", f"處理完畢！\n共生成了 {count} 張裁切影像。")
            
        except Exception as e:
            self.log(f"❌ [錯誤] 發生異常: {str(e)}")
            messagebox.showerror("錯誤", f"程式執行發生問題:\n{str(e)}")
            
        finally:
            # 重新啟用按鈕
            self.start_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = CropperGUI(root)
    root.mainloop()