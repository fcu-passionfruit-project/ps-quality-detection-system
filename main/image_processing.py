import cv2
import numpy as np

def apply_circular_mask(crop_img, mask_scale=1.01): # mask_scale=1.01是人工調適後的最佳結果
    """
    將裁切後的圖片套用圓形遮罩，遮住邊緣背景。
    
    參數:
        crop_img: 裁切後的 BGR 影像。
        mask_scale: 遮罩放大倍率。
    傳回:
        套用圓形遮罩後的 BGR 影像。
    """
    h, w = crop_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    rx = int((w // 2) * mask_scale)
    ry = int((h // 2) * mask_scale)
    
    cv2.ellipse(mask, (w // 2, h // 2), (rx, ry), 0, 0, 360, 255, -1)
    masked_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)
    return masked_img

def process_grayscale_clahe(img, clip_limit=2.0, tile_size=8):
    """
    將彩色影像轉換為灰階並套用 CLAHE (局部自適應直方圖均衡化)。
    
    參數:
        img: 輸入的 BGR 彩色影像。
        clip_limit: 對比度限制閾值。
        tile_size: 局部自適應網格的大小 (tile_size x tile_size)。
    傳回:
        處理後的單通道灰階增強影像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    return enhanced