# 影像即時串流測試程式，使用 OpenCV 開啟電腦的鏡頭，並把畫面丟給訓練好的 YOLO 模型辨識，最後把結果顯示在視窗上。

import cv2
from ultralytics import YOLO
# 1. 載入訓練好的 YOLO 模型
# 注意！這裡的路徑要改成電腦上 PassionFruitModel_v1.pt 的位置
model = YOLO("PassionFruitModel_v1.pt")

# 2. 開啟視訊鏡頭
# 數字 0 通常代表電腦的預設鏡頭 (筆電內建的)。如果有外接 USB 鏡頭，可以試試看改成 1 或 2。
cap = cv2.VideoCapture(0)

# 檢查一下鏡頭有沒有成功開啟
if not cap.isOpened():
    print("NO CAMERA!!!!!")
    exit()

print("OPEN CAMARA (按下鍵盤的 'q' 鍵可以關閉)")

# 3. 進入不斷看畫面的無窮迴圈
while True:
    # 讀取鏡頭現在的畫面 (ret 是有沒有成功讀到，frame 是照片本體)
    ret, frame = cap.read()
    
    # 如果沒讀到畫面就跳出迴圈
    if not ret:
        print("讀不到畫面，勾八")
        break

    # 4. 把剛剛拍到的畫面直接丟給 YOLO 辨識！
    # 這裡加了一個 conf=0.5，意思是「模型有 50% 以上把握才畫框框」，可以減少亂認的機率
    results = model(frame, conf=0.5)

    # 5. 把畫好框框的結果拿出來
    annotated_frame = results[0].plot()

    # 6. 顯示在一個新的視窗上
    cv2.imshow("百香果即時偵測系統 (按 Q 離開)", annotated_frame)

    # 7. 讓程式等 1 毫秒，順便偵測有沒有按下 'q' 鍵
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("高歌離席~")
        break

# 離開迴圈後，鏡頭關掉、資源釋放
cap.release()
cv2.destroyAllWindows()