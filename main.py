import cv2 
from models.traffic_light import detect_traffic_light
from models.license_plate import detect_license_plate 
from models.traffic_sign import detect_traffic_sign 
from models.vehicle import detect_vehicle
from models.stop_line import detect_stopline
from models.helmet import detect_helmet
from utils.ROI import draw_stopline_rois
from utils.traffic_violation import TrafficViolationTracker
from utils.violation_sidebar import ViolationSidebar

# Mở video 
video_path = "21s_vidnew.mov"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Không mở được video {video_path}")
raw_fps = cap.get(cv2.CAP_PROP_FPS)
fps = raw_fps if 0 < raw_fps <= 120 else 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Chuẩn hóa kích thước xử lý về chuẩn 1280x720 (tỷ lệ 16:9):
# - Giúp 6 mô hình AI chạy nhanh hơn gấp 5 lần so với video gốc 4K (3840x2160)
# - Các ngưỡng thuật toán phân tích quỹ đạo và kích thước Sidebar, font chữ luôn chuẩn xác và cân đối
width = 1280
height = 720

# Khởi tạo màn hình phụ (Sidebar) bên cạnh video hiển thị biển số xe vi phạm (Ý tưởng 4)
# sidebar_width=416 để total_width = 1280 + 416 = 1696 chia hết cho 16, đảm bảo chuẩn nén video không bị lỗi macroblock
sidebar = ViolationSidebar(sidebar_width=416, height=height, fps=fps, max_cards=4)
total_width = width + sidebar.sidebar_width

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter.fourcc(*"mp4v"),
    fps,
    (total_width, height)
)

# Khởi tạo bộ theo dõi vi phạm và phân tích quỹ đạo xe (Ý tưởng 1 & 3):
# - Ngưỡng thời gian quan sát tối đa: 3.0 giây
# - Ngưỡng lệch ngang để nhận diện rẽ: 25 pixel
# - Ngưỡng độ sâu Y tiến vào tim giao lộ để chốt vi phạm đi thẳng: 150 pixel
violation_tracker = TrafficViolationTracker(
    fps=fps,
    observe_seconds=3.0,
    turn_threshold_px=25,
    deep_threshold_px=150,
)


# Stopline khoi tao va cap nhat dong qua tung frame video
active_stopline = None

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret: 
        break 
    frame_idx += 1
    
    # Chuan hoa frame ve kich thuoc chuan 1280x720 neu video dau vao khac kich thuoc (nhu video 4K 3840x2160)
    if (frame.shape[1], frame.shape[0]) != (width, height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Chay cac model trong cung mot frame (vehicle chay che do tracking)
    stopline_result = list(detect_stopline(frame))
    traffic_light_result = list(detect_traffic_light(frame))
    helmet_result = list(detect_helmet(frame))
    license_plate_result = list(detect_license_plate(frame))
    traffic_sign_result = list(detect_traffic_sign(frame))
    vehicle_result = list(detect_vehicle(frame, track=True))

    # Cap nhat stopline dong tu model detect_stopline o frame hien tai:
    # Loc cac box nam trong khu vuc giao lo (0.35h <= y1 <= 0.65h)
    current_stopline = None
    if stopline_result[0].boxes is not None and len(stopline_result[0].boxes) > 0:
        valid_boxes = [
            b for b in stopline_result[0].boxes
            if height * 0.35 <= float(b.xyxy[0][1]) <= height * 0.65
        ]
        if len(valid_boxes) > 0:
            best_box = max(valid_boxes, key=lambda b: float(b.conf[0]))
            current_stopline = tuple(map(int, best_box.xyxy[0].tolist()))

    # Neu frame hien tai phat hien duoc stopline: cap nhat active_stopline dong
    if current_stopline is not None:
        active_stopline = current_stopline
    # Neu frame nao bi che khuat / model khong bat duoc: giu nguyen active_stopline truoc do de khong bi mat

    sl_y_top = min(active_stopline[1], active_stopline[3]) if active_stopline is not None else None

    # BUOC 1: Ve 2 vung ROI tu stopline
    top_roi, bottom_roi = None, None
    if active_stopline is not None:
        top_roi, bottom_roi = draw_stopline_rois(
            frame,
            active_stopline,
            top_label="ROI Top",
            bottom_label="ROI Bottom"
        )

    # 2. Cập nhật và phân tích vi phạm dựa trên quỹ đạo xe & tín hiệu đèn:
    # - Xe từ làn khác đi vào ROI TOP sẽ KHÔNG được nhận diện.
    # - CHỈ những xe ở ROI BOTTOM rồi đi vào bounding box của stopline thì bắt đầu theo dõi quỹ đạo trong 3s.
    violation_tracker.update(
        frame_idx=frame_idx,
        vehicle_boxes=vehicle_result[0].boxes,
        vehicle_names=vehicle_result[0].names,
        traffic_light_boxes=traffic_light_result[0].boxes,
        light_names=traffic_light_result[0].names,
        stopline_bbox=active_stopline,
        top_roi=top_roi,
        bottom_roi=bottom_roi,
    )

    # 3. Quản lý biển số xe và ghi nhận vi phạm vào Sidebar
    # - Nhận diện và lưu ảnh biển số + OCR văn bản khi xe ở gần (ROI Bottom)
    # - Khi phát hiện vi phạm, sidebar hiển thị thẻ chi tiết (ảnh + văn bản OCR + lỗi)
    #   và lưu vào vùng in danh sách biển số xe vi phạm ở dưới sidebar
    recognized_vehicle_boxes = violation_tracker.get_recognized_boxes(vehicle_result[0].boxes)
    sidebar.update_plates(
        vehicle_boxes=recognized_vehicle_boxes,
        plate_boxes=license_plate_result[0].boxes,
        frame=frame
    )
    sidebar.check_violations(violation_tracker, frame, frame_idx)

    # 4. Vẽ quỹ đạo xe, trạng thái vi phạm và bảng HUD thống kê lên frame
    frame = violation_tracker.draw_visuals(frame, frame_idx)

    # BUOC 5: Ve bouding box cua Den giao thong, Mu bao hiem, Bien bao va Bien so
    # 5.1. Ve Den giao thong (traffic light)
    if traffic_light_result[0].boxes is not None and len(traffic_light_result[0].boxes) > 0:
        for box in traffic_light_result[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = traffic_light_result[0].names[cls_id]
            # Phan mau theo trang thai den (Do / Xanh / Vang)
            if "red" in label:
                tl_color = (0, 0, 255)
            elif "green" in label:
                tl_color = (0, 255, 0)
            else:
                tl_color = (0, 215, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), tl_color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tl_color, 1, cv2.LINE_AA)

    # 5.2. Ve Mu bao hiem (helmet / no_helmet)
    if helmet_result[0].boxes is not None and len(helmet_result[0].boxes) > 0:
        for box in helmet_result[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = helmet_result[0].names[cls_id]

            # Trong ROI TOP: chi ve mu neu thuoc ve nguoi tren xe duoc cong nhan
            if sl_y_top is not None and y2 < sl_y_top:
                hm_cx = (x1 + x2) // 2
                hm_cy = (y1 + y2) // 2
                belongs_to_rec = False
                for v_box in recognized_vehicle_boxes:
                    vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0].tolist())
                    if (vx1 - 15 <= hm_cx <= vx2 + 15) and (vy1 - 15 <= hm_cy <= vy2 + 15):
                        belongs_to_rec = True
                        break
                if not belongs_to_rec:
                    continue

            hm_color = (0, 255, 128) if label == "helmet" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), hm_color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, hm_color, 1, cv2.LINE_AA)

    # 5.3. Ve Bien bao giao thong (traffic sign)
    if traffic_sign_result[0].boxes is not None and len(traffic_sign_result[0].boxes) > 0:
        for box in traffic_sign_result[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = traffic_sign_result[0].names[cls_id]
            sign_color = (255, 128, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), sign_color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sign_color, 1, cv2.LINE_AA)

    # 5.4. Ve Bien so xe (license plate)
    if license_plate_result[0].boxes is not None and len(license_plate_result[0].boxes) > 0:
        for box in license_plate_result[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = license_plate_result[0].names[cls_id]

            # Bien so trong ROI TOP: chi ve neu thuoc ve xe duoc cong nhan
            if sl_y_top is not None and y2 < sl_y_top:
                plate_cx = (x1 + x2) // 2
                plate_cy = (y1 + y2) // 2
                belongs_to_recognized = False
                for v_box in recognized_vehicle_boxes:
                    vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0].tolist())
                    if (vx1 - 20 <= plate_cx <= vx2 + 20) and (vy1 - 20 <= plate_cy <= vy2 + 20):
                        belongs_to_recognized = True
                        break
                if not belongs_to_recognized:
                    continue

            plate_color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), plate_color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, plate_color, 1, cv2.LINE_AA)

    # BUOC 6: Ve STOPLINE SAU CUNG - luon hien thi tren cung moi layer
    if active_stopline is not None:
        sx1, sy1, sx2, sy2 = active_stopline
        # Vien trang tao do tuong phan
        cv2.rectangle(frame, (sx1 - 2, sy1 - 2), (sx2 + 2, sy2 + 2), (255, 255, 255), 4)
        # Box chinh mau do dam noi bat
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
        # Nhan STOPLINE voi nen do
        stop_label = "STOPLINE"
        (tw, th), _ = cv2.getTextSize(stop_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lbl_x, lbl_y = sx1, max(sy1 - 5, th + 6)
        cv2.rectangle(frame, (lbl_x, lbl_y - th - 4), (lbl_x + tw + 10, lbl_y + 4), (0, 0, 180), -1)
        cv2.putText(frame, stop_label, (lbl_x + 5, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 7. Kết xuất màn hình phụ (Sidebar) bên cạnh video hiển thị danh sách xe vi phạm (Ý tưởng 4)
    final_canvas = sidebar.render(frame, frame_idx)

    if frame_idx % 30 == 0 or frame_idx == total_frames:
        print(
            f"Xử lý frame {frame_idx}/{total_frames} | Vi phạm vượt đèn: {violation_tracker.violation_count} | Rẽ hợp lệ: {violation_tracker.legitimate_turn_count}",
            flush=True,
        )

    # Ghi frame tổng hợp (Video + Sidebar) vào video đầu ra
    out.write(final_canvas)

# Giải phóng tài nguyên sau khi xử lý video
cap.release()
out.release()
cv2.destroyAllWindows()

print("Đã xuất video output.mp4")

