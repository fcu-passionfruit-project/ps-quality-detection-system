# 批量處理 轉換灰階影像
"""
調整 clipLimit : 預設值是 2.0。如果想要微小的皺摺陰影更明顯，可以把它拉高到 3.0 到 4.0 試試看。但不要拉得太誇張, 不然原本平滑的地方會跑出一些沙沙的雜訊(Noises)。
調整 tileGridSize:預設是 8 (代表 $8 \times 8$ 的局部網格)。
如果你發現百香果局部有些地方太亮(反光)或太暗，可以稍微加大這個數值(例如設成 16), 讓局部平衡的範圍擴大。
"""
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# ==================== 核心影像處理邏輯 ====================

def process_grayscale_clahe(img, clip_limit, tile_size):
    """
    將彩色影像轉換為灰階並套用 CLAHE 增強
    
    參數:
        img: 原始 BGR 彩色影像
        clip_limit: CLAHE 的對比度限制閾值
        tile_size: 局部自適應網格的大小 (tile_size, tile_size)
    傳回:
        處理後的灰階增強影像
    """
    # 步驟 1: 將 BGR 彩色影像轉換為單通道灰階影像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 步驟 2: 建立 CLAHE 物件並套用以強化表面紋路
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    
    return enhanced

# ==================== GUI 介面設計 ====================

class EnhancerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("百香果 ROI 灰階與紋路增強產生器")
        self.root.geometry("650x550")
        self.root.minsize(550, 450)
        
        # 設定路徑變數
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
        
        # 輸入資料夾
        ttk.Label(path_frame, text="ROI 照片來源資料夾:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(path_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="選取...", command=self.select_input).grid(row=0, column=2, padx=5)
        
        # 輸出資料夾
        ttk.Label(path_frame, text="增強照片輸出資料夾:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(path_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="選取...", command=self.select_output).grid(row=1, column=2, padx=5)
        
        # 2. 參數調整區塊 (滑條)
        param_frame = ttk.LabelFrame(self.root, text=" CLAHE 參數動態調整 ", padding=10)
        param_frame.pack(fill="x", padx=15, pady=10)
        
        # 對比限制 (clipLimit)
        ttk.Label(param_frame, text="對比限制閾值 (clipLimit):").grid(row=0, column=0, sticky="w", pady=5)
        self.clip_slider = ttk.Scale(param_frame, from_=1.0, to=10.0, value=2.0, command=self.update_labels)
        self.clip_slider.grid(row=0, column=1, sticky="we", padx=10, pady=5)
        self.clip_label = ttk.Label(param_frame, text="2.00")
        self.clip_label.grid(row=0, column=2, padx=5)
        
        # 網格大小 (tileGridSize)
        ttk.Label(param_frame, text="區域網格大小 (tileGridSize):").grid(row=1, column=0, sticky="w", pady=5)
        self.tile_slider = ttk.Scale(param_frame, from_=2, to=32, value=8, command=self.update_labels)
        self.tile_slider.grid(row=1, column=1, sticky="we", padx=10, pady=5)
        self.tile_label = ttk.Label(param_frame, text="8")
        self.tile_label.grid(row=1, column=2, padx=5)
        
        # 讓滑條欄位可以自動伸縮
        param_frame.columnconfigure(1, weight=1)
        
        # 3. 開始按鈕
        self.start_btn = ttk.Button(self.root, text="🚀 開始批量灰階與 CLAHE 處理", command=self.start_processing_thread)
        self.start_btn.pack(pady=10, ipadx=20, ipady=5)
        
        # 4. 狀態日誌區塊
        log_frame = ttk.LabelFrame(self.root, text=" 執行進度與日誌 ", padding=5)
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.log_text = ScrolledText(log_frame, height=10, bg="#f5f5f5", font=("Microsoft JhengHei", 9))
        self.log_text.pack(fill="both", expand=True)
        self.log("已準備就緒，請設定好資料夾與參數後按下開始按鈕。")

    # ==================== 按鈕觸發事件 ====================
    
    def select_input(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.input_dir.set(dir_path)
            
    def select_output(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)
            
    def update_labels(self, event=None):
        """即時將滑條數值更新至顯示標籤"""
        self.clip_label.config(text=f"{self.clip_slider.get():.2f}")
        self.tile_label.config(text=f"{int(self.tile_slider.get())}")

    def start_processing_thread(self):
        """使用多執行緒啟動後台處理，避免 GUI 視窗凍結"""
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showwarning("提示", "來源資料夾與輸出資料夾尚未選填完整！")
            return
        
        self.start_btn.config(state="disabled")
        
        t = threading.Thread(target=self.process_core)
        t.daemon = True
        t.start()

    # ==================== 後台批次處理核心 ====================

    def process_core(self):
        try:
            in_dir = self.input_dir.get()
            out_dir = self.output_dir.get()
            
            clip_val = float(self.clip_slider.get())
            tile_val = int(self.tile_slider.get())
            
            os.makedirs(out_dir, exist_ok=True)
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            img_files = [f for f in os.listdir(in_dir) if f.lower().endswith(valid_extensions)]
            
            self.log(f"\n[系統] 找到 {len(img_files)} 張影像，開始進行批量處理...")
            
            count = 0
            for current_idx, img_name in enumerate(img_files, 1):
                img_path = os.path.join(in_dir, img_name)
                
                # 解決 OpenCV 無法直接讀取含有中文或特殊字元路徑的替代方案
                raw_data = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                # 執行灰階與 CLAHE 影像處理
                processed_img = process_grayscale_clahe(img, clip_limit=clip_val, tile_size=tile_val)
                
                # 解決 OpenCV 無法直接儲存含有中文或特殊字元路徑的替代方案
                save_path = os.path.join(out_dir, img_name)
                ext = os.path.splitext(img_name)[1]
                success, img_encode = cv2.imencode(ext, processed_img)
                if success:
                    img_encode.tofile(save_path)
                    count += 1
                
                if current_idx % 10 == 0 or current_idx == len(img_files):
                    self.log(f"[進度] 已處理 {current_idx}/{len(img_files)} 張影像...")
            
            self.log(f"🎉 [成功] 轉換完成，共處理 {count} 張百香果 ROI 照片。")
            messagebox.showinfo("完成", f"處理完畢！\n已成功轉換並輸出 {count} 張增強影像。")
            
        except Exception as e:
            self.log(f"❌ [錯誤] 發生異常: {str(e)}")
            messagebox.showerror("錯誤", f"程式執行發生問題:\n{str(e)}")
            
        finally:
            self.start_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancerGUI(root)
    root.mainloop()