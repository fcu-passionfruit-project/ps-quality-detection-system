import cv2
from ultralytics import YOLO

class PassionFruitModelManager:
    """
    管理百香果多階段辨識系統中所有 YOLO 模型的載入與推論。
    """
    def __init__(self, stage1_path="stage1.pt", stage2_1_path="stage2_1.pt", stage2_2_path="stage2_2.pt"):
        """
        初始化並載入三個階段的 YOLO 模型。
        """
        self.model_stage1 = YOLO(stage1_path)
        self.model_stage2_1 = YOLO(stage2_1_path)
        self.model_stage2_2 = YOLO(stage2_2_path)

    def detect_stage1(self, file_path):
        """
        執行第一階段百香果目標偵測。
        """
        return self.model_stage1(file_path)[0]

    def classify_color(self, img_bgr):
        """
        執行第二階段_1 顏色分類推論。
        """
        return self.model_stage2_1(img_bgr)[0]

    def classify_wrinkle(self, img_gray):
        """
        執行第二階段_2 皺褶分類推論。將單通道灰階轉為三通道以符合模型輸入需求。
        """
        clahe_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        return self.model_stage2_2(clahe_bgr)[0]