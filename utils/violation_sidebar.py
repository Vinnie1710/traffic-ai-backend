# Module quản lý danh sách xe vi phạm, trích xuất ảnh biển số, OCR văn bản và render thanh giám sát (Sidebar) bên cạnh video
# Đáp ứng yêu cầu:
# 1. Nhận diện biển số xe -> hiện kết quả ở sidebar gồm: Ảnh biển số, Biển số dưới dạng văn bản (OCR), Vi phạm
# 2. Tạo một vùng nằm trong sidebar ở dưới để lưu và in ra biển số xe vi phạm

import cv2
import numpy as np
from models.ocr import recognize_plate_text
from models.license_plate import model as lp_yolo_model


class ViolationSidebar:
    """
    Quản lý danh sách phương tiện vi phạm và kết xuất giao diện thanh bên (Sidebar)
    hiển thị ảnh crop biển số xe, văn bản biển số nhận diện bằng OCR, thời điểm và loại vi phạm,
    đồng thời duy trì vùng lưu trữ và in ra danh sách các biển số xe vi phạm ở nửa dưới sidebar.
    """

    def __init__(self, sidebar_width=416, height=720, fps=30.0, max_cards=2):
        self.sidebar_width = int(sidebar_width)
        self.height = int(height)
        self.fps = max(float(fps), 1.0)
        self.max_cards = int(max_cards)

        # Bộ nhớ lưu ảnh biển số tốt nhất và văn bản OCR của từng xe:
        # {track_id: (best_crop, conf, plate_bbox, area, plate_text)}
        self.vehicle_plates = {}

        # Danh sách sự kiện vi phạm đã ghi nhận:
        # [ {
        #     'track_id': int,
        #     'cls_name': str,
        #     'plate_crop': np.ndarray,
        #     'plate_text': str,
        #     'time_str': str,
        #     'frame_idx': int,
        #     'reason': str,
        # } ]
        self.violations_history = []
        self.recorded_violation_ids = set()

    def update_plates(self, vehicle_boxes, plate_boxes, frame):
        """
        Ghép nối (associate) bounding box biển số với xe tương ứng và lưu lại ảnh crop đẹp nhất
        kèm kết quả OCR biển số xe dạng văn bản.

        Chiến lược:
        1. Ưu tiên ghép từ plate_boxes (toàn frame).
        2. Nếu xe ở gần (ROI Bottom) chưa bắt được biển từ toàn frame, crop vùng đuôi xe (nửa dưới xe)
           rồi chạy detector biển số trực tiếp trên sub-crop để bắt được biển số xe máy nhỏ.
        3. Sử dụng mô hình OCR-26s để đọc văn bản biển số xe.
        """
        if vehicle_boxes is None or len(vehicle_boxes) == 0:
            return

        h, w = frame.shape[:2]

        for v_box in vehicle_boxes:
            if v_box.id is None:
                continue
            track_id = int(v_box.id[0])
            vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0].tolist())
            car_w = vx2 - vx1
            car_h = vy2 - vy1

            found_crop = None
            found_conf = 0.0
            found_bbox = None

            # Cách 1: Tìm xem có biển số nào từ plate_boxes nằm trong phạm vi xe này không
            if plate_boxes is not None and len(plate_boxes) > 0:
                for p_box in plate_boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                    p_conf = float(p_box.conf[0])
                    p_cx = (px1 + px2) // 2
                    p_cy = (py1 + py2) // 2

                    if (vx1 - 15) <= p_cx <= (vx2 + 15) and (vy1 - 15) <= p_cy <= (vy2 + 15):
                        pad = 3
                        cpy1 = max(0, py1 - pad)
                        cpy2 = min(h, py2 + pad)
                        cpx1 = max(0, px1 - pad)
                        cpx2 = min(w, px2 + pad)
                        if cpy2 > cpy1 and cpx2 > cpx1:
                            crop = frame[cpy1:cpy2, cpx1:cpx2].copy()
                            found_crop = crop
                            found_conf = p_conf
                            found_bbox = (px1, py1, px2, py2)
                            break

            # Cách 2: Nếu chưa tìm thấy và xe chưa có biển số trong cache, crop vùng đuôi xe và detect
            if found_crop is None and track_id not in self.vehicle_plates:
                sub_y1 = max(0, int(vy1 + 0.35 * car_h))
                sub_y2 = min(h, vy2)
                sub_x1 = max(0, vx1)
                sub_x2 = min(w, vx2)
                sub_car = frame[sub_y1:sub_y2, sub_x1:sub_x2]
                if sub_car.size > 0:
                    try:
                        lp_res = lp_yolo_model.predict(sub_car, conf=0.12, verbose=False)
                        if len(lp_res[0].boxes) > 0:
                            best_b = max(lp_res[0].boxes, key=lambda b: float(b.conf[0]))
                            bx1, by1, bx2, by2 = map(int, best_b.xyxy[0].tolist())
                            crop = sub_car[by1:by2, bx1:bx2].copy()
                            if crop.size > 0:
                                found_crop = crop
                                found_conf = float(best_b.conf[0])
                                found_bbox = (sub_x1 + bx1, sub_y1 + by1, sub_x1 + bx2, sub_y1 + by2)
                    except Exception:
                        pass

            # Nếu tìm được ảnh crop biển số, đánh giá và cập nhật nếu tốt hơn
            if found_crop is not None and found_crop.size > 0:
                p_area = found_crop.shape[0] * found_crop.shape[1]
                should_update = False

                if track_id not in self.vehicle_plates:
                    should_update = True
                else:
                    prev_area = self.vehicle_plates[track_id][3]
                    prev_conf = self.vehicle_plates[track_id][1]
                    # Cập nhật nếu biển to hơn (ở gần camera hơn) hoặc cùng cỡ nhưng nét hơn
                    if p_area > prev_area or (p_area >= prev_area * 0.9 and found_conf > prev_conf):
                        should_update = True

                if should_update:
                    # Chạy mô hình OCR để nhận diện chuỗi văn bản biển số
                    ocr_text = recognize_plate_text(found_crop, conf_thresh=0.10)
                    self.vehicle_plates[track_id] = (
                        found_crop,
                        found_conf,
                        found_bbox,
                        p_area,
                        ocr_text,
                    )

    def check_violations(self, violation_tracker, frame, frame_idx):
        """
        Quét trạng thái từ violation_tracker, khi phát hiện xe vi phạm mới thì đưa vào danh sách hiển thị
        và tự động lưu vào danh sách nhật ký biển số xe vi phạm.
        """
        h, w = frame.shape[:2]

        for track_id, v_info in violation_tracker.vehicles.items():
            if v_info.get("status") == "VIOLATION" and track_id not in self.recorded_violation_ids:
                self.recorded_violation_ids.add(track_id)

                cls_name = v_info.get("class_name", "xe")
                reason = v_info.get("reason", "Vượt đèn đỏ")
                v_bbox = v_info.get("bbox")

                plate_crop = None
                plate_text = ""

                # 1. Lấy từ bộ nhớ cache đã nhận diện khi xe ở gần (ROI Bottom)
                if track_id in self.vehicle_plates:
                    plate_crop = self.vehicle_plates[track_id][0]
                    plate_text = self.vehicle_plates[track_id][4]

                # 2. Fallback cứu cánh: Nếu xe chưa có plate_crop, thử crop vùng đuôi xe tại frame hiện tại
                if plate_crop is None and v_bbox is not None:
                    bx1, by1, bx2, by2 = v_bbox
                    sub_y1 = max(0, int(by1 + 0.4 * (by2 - by1)))
                    sub_y2 = min(h, by2)
                    sub_x1 = max(0, bx1)
                    sub_x2 = min(w, bx2)
                    if sub_y2 > sub_y1 and sub_x2 > sub_x1:
                        sub_car = frame[sub_y1:sub_y2, sub_x1:sub_x2]
                        try:
                            lp_res = lp_yolo_model.predict(sub_car, conf=0.10, verbose=False)
                            if len(lp_res[0].boxes) > 0:
                                best_b = max(lp_res[0].boxes, key=lambda b: float(b.conf[0]))
                                pbx1, pby1, pbx2, pby2 = map(int, best_b.xyxy[0].tolist())
                                plate_crop = sub_car[pby1:pby2, pbx1:pbx2].copy()
                        except Exception:
                            pass
                        if plate_crop is None:
                            plate_crop = sub_car.copy()

                    if plate_crop is not None and plate_crop.size > 0:
                        plate_text = recognize_plate_text(plate_crop, conf_thresh=0.10)

                # 3. Chuẩn hóa biển số dưới dạng văn bản
                # Nếu OCR đọc được ký tự rõ ràng thì dùng OCR text, nếu quá mờ hoặc rỗng thì định dạng theo chuẩn
                if not plate_text or len(plate_text.strip()) < 2:
                    clean_text = f"XE-{cls_name.upper()}-{track_id:03d}"
                else:
                    clean_text = plate_text.strip().upper()

                # 4. Tính thời gian vi phạm (phút : giây . mili-giây)
                time_sec = frame_idx / self.fps
                minutes = int(time_sec // 60)
                seconds = time_sec % 60
                time_str = f"{minutes:02d}:{seconds:04.1f}s"

                # Thêm vào đầu danh sách vi phạm (mới nhất lên trên cùng)
                self.violations_history.insert(0, {
                    "track_id": track_id,
                    "cls_name": cls_name,
                    "plate_crop": plate_crop,
                    "plate_text": clean_text,
                    "time_str": time_str,
                    "frame_idx": frame_idx,
                    "reason": reason,
                })

    def render(self, frame, frame_idx):
        """
        Ghép frame video gốc (bên trái) với thanh Sidebar hiển thị xe vi phạm (bên phải).
        Bao gồm:
        - Phía trên: Thẻ chi tiết xe vi phạm (Ảnh biển số, Biển số dạng văn bản OCR, Loại vi phạm).
        - Phía dưới: Vùng lưu trữ và in ra danh sách biển số xe vi phạm.
        """
        h, w = frame.shape[:2]
        total_w = w + self.sidebar_width

        # Canvas tổng hợp (W + sidebar_width) x H
        canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        sb_x1 = w
        sb_x2 = total_w

        # Nền Sidebar màu xám đậm hiện đại
        cv2.rectangle(canvas, (sb_x1, 0), (sb_x2, h), (24, 22, 28), -1)
        # Đường viền phân cách giữa video và sidebar
        cv2.line(canvas, (sb_x1, 0), (sb_x1, h), (65, 65, 80), 2)

        # ----------------- 1. HEADER SIDEBAR -----------------
        header_h = 70
        cv2.rectangle(canvas, (sb_x1, 0), (sb_x2, header_h), (35, 30, 40), -1)
        cv2.line(canvas, (sb_x1, header_h), (sb_x2, header_h), (0, 0, 220), 2)

        # Icon chấm đỏ nhấp nháy cảnh báo
        cv2.circle(canvas, (sb_x1 + 25, 26), 7, (0, 0, 255), -1)
        cv2.circle(canvas, (sb_x1 + 25, 26), 11, (0, 0, 255), 1)

        cv2.putText(
            canvas,
            "HE THONG GIAM SAT PHAT NGUOI",
            (sb_x1 + 45, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        num_violations = len(self.violations_history)
        cv2.putText(
            canvas,
            f"DANH SACH VI PHAM: {num_violations} PHUONG TIEN",
            (sb_x1 + 45, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

        # ----------------- 2. KHU VỰC TRÊN: THẺ CHI TIẾT VI PHẠM (ẢNH BIỂN SỐ, VĂN BẢN, VI PHẠM) -----------------
        top_section_y1 = header_h + 10
        top_section_y2 = 430
        card_h = 160
        card_gap = 10

        if len(self.violations_history) == 0:
            # Trạng thái chờ khi chưa phát hiện vi phạm
            box_y1 = top_section_y1 + 40
            box_y2 = box_y1 + 180
            bx1, bx2 = sb_x1 + 20, sb_x2 - 20
            cv2.rectangle(canvas, (bx1, box_y1), (bx2, box_y2), (32, 28, 36), -1)
            cv2.rectangle(canvas, (bx1, box_y1), (bx2, box_y2), (55, 55, 70), 1)

            cv2.circle(canvas, (sb_x1 + self.sidebar_width // 2, box_y1 + 55), 24, (0, 180, 0), 2)
            cv2.putText(
                canvas,
                "OK",
                (sb_x1 + self.sidebar_width // 2 - 12, box_y1 + 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                "DANG QUAN SAT GIAO THONG",
                (sb_x1 + 45, box_y1 + 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Chua phat hien phuong tien vi pham",
                (sb_x1 + 35, box_y1 + 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )
        else:
            # Hiển thị các thẻ vi phạm mới nhất
            curr_y = top_section_y1
            for idx, item in enumerate(self.violations_history[:self.max_cards]):
                cx1 = sb_x1 + 12
                cx2 = sb_x2 - 12
                cy1 = curr_y
                cy2 = curr_y + card_h

                # Nền thẻ vi phạm
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (35, 32, 42), -1)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 0, 220), 1)

                # Vạch đỏ bên trái thẻ
                cv2.rectangle(canvas, (cx1, cy1), (cx1 + 5, cy2), (0, 0, 255), -1)

                # Badge "VI PHAM"
                badge_w, badge_h = 90, 20
                cv2.rectangle(canvas, (cx2 - badge_w - 6, cy1 + 8), (cx2 - 6, cy1 + 8 + badge_h), (0, 0, 180), -1)
                cv2.putText(
                    canvas,
                    "VI PHAM",
                    (cx2 - badge_w + 10, cy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # 1. ẢNH BIỂN SỐ XE (Bên trái thẻ)
                plate_crop = item.get("plate_crop")
                target_pw, target_ph = 130, 58
                crop_px1 = cx1 + 14
                crop_py1 = cy1 + 18
                crop_px2 = crop_px1 + target_pw
                crop_py2 = crop_py1 + target_ph

                if plate_crop is not None and plate_crop.size > 0:
                    try:
                        resized_crop = cv2.resize(plate_crop, (target_pw, target_ph), interpolation=cv2.INTER_LINEAR)
                        canvas[crop_py1:crop_py2, crop_px1:crop_px2] = resized_crop
                    except Exception:
                        pass
                else:
                    cv2.rectangle(canvas, (crop_px1, crop_py1), (crop_px2, crop_py2), (50, 50, 60), -1)
                    cv2.putText(
                        canvas,
                        "NO IMAGE",
                        (crop_px1 + 20, crop_py1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (150, 150, 150),
                        1,
                    )

                # Viền neon vàng bọc ảnh biển số
                cv2.rectangle(canvas, (crop_px1, crop_py1), (crop_px2, crop_py2), (0, 215, 255), 1)
                # Nhãn phụ nhỏ dưới ảnh biển số
                cv2.putText(
                    canvas,
                    "[ANH BIEN SO]",
                    (crop_px1 + 18, crop_py2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA,
                )

                # 2. BIỂN SỐ DƯỚI DẠNG VĂN BẢN (TEXT OCR) (Bên phải thẻ)
                info_x = crop_px2 + 14
                cv2.putText(
                    canvas,
                    "BIEN SO (OCR):",
                    (info_x, cy1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                # Chữ văn bản biển số to rõ màu vàng chanh nổi bật
                cv2.putText(
                    canvas,
                    f"{item['plate_text']}",
                    (info_x, cy1 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Thông tin loại xe và ID
                cv2.putText(
                    canvas,
                    f"Xe: {item['cls_name'].upper()} (ID: {item['track_id']})",
                    (info_x, cy1 + 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                # 3. LỖI VI PHẠM (Dòng cảnh báo đỏ tươi)
                cv2.putText(
                    canvas,
                    f"Loi: {item['reason']}",
                    (cx1 + 14, cy1 + 118),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (80, 80, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Thời điểm vi phạm & Badge "DA GHI HINH"
                cv2.putText(
                    canvas,
                    f"Thoi gian: {item['time_str']}",
                    (cx1 + 14, cy1 + 142),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    canvas,
                    "[DA GHI HINH]",
                    (cx2 - 105, cy1 + 142),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 128),
                    1,
                    cv2.LINE_AA,
                )

                curr_y += card_h + card_gap

        # ----------------- 3. KHU VỰC DƯỚI: VÙNG LƯU VÀ IN RA BIỂN SỐ XE VI PHẠM (LOG PANEL) -----------------
        log_y1 = 430
        log_y2 = h - 25
        log_x1 = sb_x1 + 10
        log_x2 = sb_x2 - 10

        # Nền hộp lưu trữ danh sách biển số vi phạm
        cv2.rectangle(canvas, (log_x1, log_y1), (log_x2, log_y2), (30, 26, 34), -1)
        cv2.rectangle(canvas, (log_x1, log_y1), (log_x2, log_y2), (80, 70, 95), 1)

        # Thanh tiêu đề vùng lưu trữ (Header Log Panel)
        header_log_h = 32
        cv2.rectangle(canvas, (log_x1, log_y1), (log_x2, log_y1 + header_log_h), (45, 38, 52), -1)
        cv2.line(canvas, (log_x1, log_y1 + header_log_h), (log_x2, log_y1 + header_log_h), (0, 165, 255), 2)

        # Icon thư mục / ổ đĩa lưu trữ
        cv2.circle(canvas, (log_x1 + 16, log_y1 + 16), 5, (0, 215, 255), -1)
        cv2.putText(
            canvas,
            "DANH SACH BIEN SO XE VI PHAM (LOG)",
            (log_x1 + 30, log_y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Tổng số xe đã ghi nhận
        cv2.putText(
            canvas,
            f"[{num_violations}]",
            (log_x2 - 35, log_y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

        # Thanh tiêu đề các cột của bảng in biển số
        col_header_y = log_y1 + header_log_h + 18
        cv2.putText(canvas, "STT", (log_x1 + 10, col_header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "BIEN SO (TEXT)", (log_x1 + 45, col_header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LOAI XE", (log_x1 + 190, col_header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "THOI GIAN", (log_x1 + 295, col_header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 180), 1, cv2.LINE_AA)

        cv2.line(canvas, (log_x1 + 5, col_header_y + 6), (log_x2 - 5, col_header_y + 6), (50, 45, 60), 1)

        # In danh sách các biển số xe vi phạm đã lưu
        row_y = col_header_y + 24
        row_h = 24
        max_rows = 6

        if len(self.violations_history) == 0:
            cv2.putText(
                canvas,
                "[ CHUA CO BIEN SO VI PHAM DUOC GHI NHAN ]",
                (log_x1 + 25, log_y1 + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (100, 100, 120),
                1,
                cv2.LINE_AA,
            )
        else:
            # In ra các xe đã vi phạm (hiển thị danh sách từ mới nhất hoặc theo thứ tự phát hiện)
            # Dùng thứ tự STT chuẩn
            for i, item in enumerate(self.violations_history[:max_rows]):
                stt_num = len(self.violations_history) - i
                row_bg = (38, 34, 44) if i % 2 == 0 else (30, 26, 34)
                cv2.rectangle(canvas, (log_x1 + 5, row_y - 14), (log_x2 - 5, row_y + 6), row_bg, -1)

                # Cột STT
                cv2.putText(
                    canvas,
                    f"#{stt_num:02d}",
                    (log_x1 + 10, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Cột Biển số dạng văn bản (nổi bật)
                cv2.putText(
                    canvas,
                    f"{item['plate_text']}",
                    (log_x1 + 45, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Cột Loại xe
                cls_str = item['cls_name'][:9].upper()
                cv2.putText(
                    canvas,
                    f"{cls_str}",
                    (log_x1 + 190, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                # Cột Thời gian vi phạm
                cv2.putText(
                    canvas,
                    f"{item['time_str']}",
                    (log_x1 + 295, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 128),
                    1,
                    cv2.LINE_AA,
                )

                row_y += row_h

        # ----------------- 4. FOOTER HỆ THỐNG -----------------
        cv2.putText(
            canvas,
            "TRAFFIC AI - HE THONG PHAT NGUOI 4.0",
            (sb_x1 + 35, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (100, 100, 120),
            1,
            cv2.LINE_AA,
        )

        return canvas
