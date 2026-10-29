from ultralytics import YOLO
import cv2
import numpy as np
import os

# Khởi tạo mô hình OCR nhận diện ký tự biển số xe
weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "OCR-26s-best.pt")
ocr_model = YOLO(weights_path)


def recognize_plate_text(plate_img, conf_thresh=0.10):
    """
    Nhận diện văn bản biển số xe từ ảnh crop biển số sử dụng mô hình OCR-26s.

    Args:
        plate_img (np.ndarray): Ảnh crop biển số xe.
        conf_thresh (float): Ngưỡng confidence tối thiểu cho ký tự.

    Returns:
        str: Chuỗi văn bản biển số xe đã được định dạng (VD: '59X1-6789' hoặc '29A-12345').
    """
    if plate_img is None or plate_img.size == 0:
        return ""

    h, w = plate_img.shape[:2]
    if h < 5 or w < 5:
        return ""

    try:
        # Predict ký tự từ mô hình OCR
        results = ocr_model.predict(
            source=plate_img,
            conf=conf_thresh,
            verbose=False,
        )
    except Exception:
        return ""

    if results is None or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return ""

    boxes = results[0].boxes
    chars = []
    for b in boxes:
        cls_id = int(b.cls[0])
        char = ocr_model.names.get(cls_id, "")
        if not char:
            continue
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        chars.append((cx, cy, char, conf))

    if len(chars) == 0:
        return ""

    # Biển số xe máy Việt Nam thường có 2 dòng (dòng trên: tỉnh/seri, dòng dưới: số)
    # Biển số ô tô có thể 1 dòng dài hoặc 2 dòng
    y_coords = [c[1] for c in chars]
    y_min, y_max = min(y_coords), max(y_coords)
    y_diff = y_max - y_min

    # Nếu độ chênh lệch Y lớn hơn 25% chiều cao biển số -> phân thành 2 dòng
    if y_diff >= 0.25 * h:
        mid_y = (y_min + y_max) / 2.0
        line1 = [c for c in chars if c[1] < mid_y]
        line2 = [c for c in chars if c[1] >= mid_y]

        line1.sort(key=lambda c: c[0])
        line2.sort(key=lambda c: c[0])

        t1 = "".join([c[2] for c in line1])
        t2 = "".join([c[2] for c in line2])

        if t1 and t2:
            return f"{t1}-{t2}"
        return t1 or t2
    else:
        # 1 dòng: sắp xếp từ trái qua phải theo X
        chars.sort(key=lambda c: c[0])
        return "".join([c[2] for c in chars])

