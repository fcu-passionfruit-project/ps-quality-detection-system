# 單張照片辨識測試程式
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# 1. 載入訓練好的 YOLO 模型
# 注意！這裡的路徑要改成電腦上 PF.pt 的位置
model = YOLO("stage1.pt")

# 2. 定義開啟照片並辨識的功能
def open_and_predict():
    # 彈出視窗讓人選擇照片
    file_path = filedialog.askopenfilename(
        title="請選擇一張百香果照片",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    # 如果按了取消沒選照片，就什麼都不做
    if not file_path:
        return

    # 讓 YOLO 進行辨識
    # 加上 conf 和 iou 參數來控制模型的嚴格度->如使用YOLO26 將沒有iou選項!->YOLO26移除NMS
    results = model(file_path)
    
    # 取出畫好框框的圖片陣列 (YOLO 預設是 BGR 顏色格式)
    im_array = results[0].plot()
    
    # 將 BGR 轉換回正常的 RGB 顏色，並轉成 PIL 圖片格式
    im_rgb = im_array[..., ::-1]
    img = Image.fromarray(im_rgb)

    # 為了怕照片太大把螢幕撐爆，把它等比例縮小一下
    img.thumbnail((1600, 1200))
    
    # 轉換成 Tkinter 看得懂的圖片格式
    img_tk = ImageTk.PhotoImage(img)

    # 把圖片塞進 GUI 的畫布標籤裡
    image_label.config(image=img_tk)
    image_label.image = img_tk  # 這行超級重要！不加的話圖片會被記憶體當垃圾清掉而顯示空白喔

# 3. 建立 Tkinter 視窗介面
root = tk.Tk()
root.title("百香果物件偵測系統")
root.geometry("1920x1080")  # 設定視窗大小
root.config(bg="#f0f0f0") # 給個溫柔的灰色背景

# 建立一個按鈕
btn = tk.Button(
    root, 
    text="點我選擇百香果照片 📁", 
    command=open_and_predict, 
    font=("微軟正黑體", 16, "bold"),
    bg="#ffdab9", # 淡淡的蜜桃色
    padx=20, pady=10
)
btn.pack(pady=20) # 把按鈕放到畫面上，並在上下留點空白

# 建立一個用來顯示圖片的區域 (一開始是空的)
image_label = tk.Label(root, bg="#f0f0f0")
image_label.pack(expand=True)

# 4. 啟動視窗！
if __name__ == "__main__":
    root.mainloop()