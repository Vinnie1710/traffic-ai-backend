# Module quản lý, theo dõi quỹ đạo xe và phát hiện vi phạm vượt đèn đỏ
# Hỗ trợ Ý tưởng 3: Thiết lập ngưỡng thời gian quan sát quỹ đạo xe sau khi qua vạch dừng
# để phân biệt xe đi thẳng (vượt đèn đỏ) và xe rẽ phải/trái hợp lệ khi có đèn rẽ nhánh.

import cv2
import numpy as np
from collections import defaultdict, deque
from utils.ROI import (
    is_point_in_roi,
    get_roi_bbox,
    is_in_roi,
    is_in_stopline_bbox,
    check_bbox_intersect,
    check_vehicle_crossed_stopline,
    check_vehicle_passed_stopline,
)


class VehicleStatus:
    APPROACHING = "APPROACHING"      # Xe đang tiếp cận trước vạch dừng
    OBSERVING = "OBSERVING"          # Đang trong ngưỡng thời gian quan sát quỹ đạo
    VIOLATION = "VIOLATION"          # Xác nhận vi phạm vượt đèn đỏ
    LEGITIMATE = "LEGITIMATE"        # Xác nhận rẽ hợp lệ theo đèn tín hiệu nhánh
    LEGAL_PASS = "LEGAL_PASS"        # Qua vạch khi đèn xanh thẳng hợp lệ
    IGNORED = "IGNORED"              # Xe trong ROI TOP không qua stopline (không nhận diện)


def parse_traffic_lights(traffic_light_boxes, class_names, conf_thresh=0.25):
    """
    Phân tích kết quả phát hiện đèn giao thông thành trạng thái boolean có cấu trúc.
    
    Args:
        traffic_light_boxes: ultralytics Boxes object hoặc list các box
        class_names: dict ánh xạ class_id -> class_name
        conf_thresh: ngưỡng tin cậy tối thiểu

    Returns:
        dict: {
            'red_light': bool,
            'green_light': bool,
            'green_right_arrow': bool,
            'green_left_arrow': bool,
            'red_left_arrow': bool,
            'yellow_light': bool,
            'active_labels': list[str]
        }
    """
    state = {
        "red_light": False,
        "green_light": False,
        "green_right_arrow": False,
        "green_left_arrow": False,
        "red_left_arrow": False,
        "yellow_light": False,
        "yellow_left_arrow": False,
        "active_labels": [],
    }

    if traffic_light_boxes is None or len(traffic_light_boxes) == 0:
        return state

    for box in traffic_light_boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, "")
        if label in state:
            state[label] = True
            if label not in state["active_labels"]:
                state["active_labels"].append(label)

    return state


class TrafficViolationTracker:
    def __init__(
        self,
        fps=30.0,
        observe_seconds=3.0,
        turn_threshold_px=25,
        deep_threshold_px=150,
        history_len=90,
    ):
        """
        Khởi tạo hệ thống theo dõi vi phạm và phân tích quỹ đạo xe.

        Args:
            fps (float): Tốc độ khung hình của video.
            observe_seconds (float): Ngưỡng thời gian quan sát tối đa (mặc định 3.0s).
            turn_threshold_px (int): Độ lệch ngang tối thiểu (pixel) để xác định xe rẽ phải hoặc rẽ trái (25px).
            deep_threshold_px (int): Độ sâu Y tiến vào tim giao lộ (pixel) để xác định xe đi thẳng vượt đèn đỏ (150px).
            history_len (int): Số frame tối đa lưu trữ lịch sử di chuyển của mỗi xe.
        """
        self.fps = max(float(fps), 1.0)
        self.observe_seconds = float(observe_seconds)
        self.max_observe_frames = max(int(self.fps * self.observe_seconds), 20)
        self.turn_threshold_px = int(turn_threshold_px)
        self.deep_threshold_px = int(deep_threshold_px)
        self.history_len = int(history_len)

        # Lịch sử tọa độ tâm đáy xe CHỈ trong vùng ROI TOP (khi đã qua stopline):
        # {track_id: deque([(cx, cy, frame_idx), ...])}
        self.trajectories = defaultdict(lambda: deque(maxlen=self.history_len))

        # Lưu vị trí frame trước của mọi xe để phát hiện thời điểm cắt qua stopline:
        self.last_positions = {}

        # Thông tin trạng thái của từng xe:
        # {track_id: {
        #     'status': VehicleStatus,
        #     'class_name': str,
        #     'start_frame': int,
        #     'start_pos': (cx, cy),
        #     'lights_at_crossing': dict,
        #     'direction': str,       # 'STRAIGHT', 'RIGHT', 'LEFT'
        #     'reason': str,
        #     'resolved_frame': int
        # }}
        self.vehicles = {}

        # Thống kê tổng hợp
        self.violation_count = 0
        self.legitimate_turn_count = 0   # rẽ phải + rẽ trái hợp lệ
        self.legitimate_right_count = 0
        self.legitimate_left_count = 0
        self.straight_legal_count = 0
        self.violation_ids = set()
        self.legitimate_ids = set()

    def is_vehicle_recognized(self, track_id):
        """
        Kiểm tra xem xe có đang được nhận diện / hiển thị hay không:
        - Các phương tiện đi từ làn khác đi vào bounding box của ROI TOP: KHÔNG nhận diện.
        - CHỈ những xe ở ROI BOTTOM rồi đi vào bounding box của stopline: được nhận diện và theo dõi quỹ đạo.
        """
        if track_id not in self.vehicles:
            return False
        return bool(self.vehicles[track_id].get("is_recognized", False))

    def get_recognized_boxes(self, vehicle_boxes):
        """
        Lọc danh sách bounding boxes của xe chỉ giữ lại các xe đang được công nhận nhận diện.
        """
        if vehicle_boxes is None or len(vehicle_boxes) == 0:
            return []
        recognized = []
        for box in vehicle_boxes:
            if box.id is not None and self.is_vehicle_recognized(int(box.id[0])):
                recognized.append(box)
        return recognized

    def get_bottom_roi_boxes(self, vehicle_boxes):
        """
        Lọc danh sách bounding boxes chỉ giữ lại các xe ĐANG Ở ROI BOTTOM (chưa qua stopline).

        Đây là tập xe gần camera nhất → biển số rõ nhất → dùng để capture ảnh biển tốt nhất
        lưu vào sidebar trước khi xe đi vào giao lộ và biển trở nên mờ hơn.

        Điều kiện:
        - Xe được công nhận nhận diện (is_recognized = True)
        - Xe đã từng ở ROI BOTTOM (was_in_bottom = True)
        - Xe CHƯA qua stopline (passed_stopline = False) → đang tiếp cận, còn gần camera
        """
        if vehicle_boxes is None or len(vehicle_boxes) == 0:
            return []
        bottom_boxes = []
        for box in vehicle_boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            v_info = self.vehicles.get(track_id)
            if v_info is None:
                continue
            if (
                v_info.get("is_recognized", False)
                and v_info.get("was_in_bottom", False)
                and not v_info.get("passed_stopline", False)
            ):
                bottom_boxes.append(box)
        return bottom_boxes

    def update(
        self,
        frame_idx,
        vehicle_boxes,
        vehicle_names,
        traffic_light_boxes,
        light_names,
        stopline_bbox,
        top_roi=None,
        bottom_roi=None,
    ):
        """
        Cập nhật vị trí các xe, trạng thái đèn và phân tích quỹ đạo trong frame hiện tại.
        Quy tắc:
        1. Các phương tiện đi từ làn khác đi vào bounding box của ROI TOP sẽ KHÔNG nhận diện phương tiện đó.
        2. CHỈ những xe ở ROI BOTTOM rồi đi vào bounding box của stopline thì bắt đầu theo dõi quỹ đạo xe
           trong ngưỡng thời gian là 3 giây.
        """
        light_state = parse_traffic_lights(traffic_light_boxes, light_names)

        # Lấy bounding box của các vùng ROI và Stopline
        top_roi_bbox = get_roi_bbox(top_roi)
        bottom_roi_bbox = get_roi_bbox(bottom_roi)

        sl_y1 = min(stopline_bbox[1], stopline_bbox[3]) if stopline_bbox is not None else None
        sl_y2 = max(stopline_bbox[1], stopline_bbox[3]) if stopline_bbox is not None else None

        current_active_ids = set()

        if vehicle_boxes is not None and len(vehicle_boxes) > 0:
            for box in vehicle_boxes:
                # Lấy track_id của xe
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                current_active_ids.add(track_id)

                cls_id = int(box.cls[0])
                vehicle_cls_name = vehicle_names.get(cls_id, "xe")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Tâm đáy xe (điểm tiếp xúc với mặt đường)
                cx = (x1 + x2) // 2
                cy = y2

                # Vị trí frame trước của xe (dùng để kiểm tra cắt qua Stopline)
                prev_pos = self.last_positions.get(track_id, None)
                self.last_positions[track_id] = (cx, cy)

                # 1. Xác định xe hiện tại có đang nằm trong ROI TOP (polygon hoặc bounding box) hay không
                in_top_roi = False
                if top_roi is not None:
                    in_top_roi = is_in_roi((x1, y1, x2, y2), top_roi, method="bottom_center")
                elif sl_y1 is not None:
                    in_top_roi = (cy < sl_y1)

                if not in_top_roi and top_roi_bbox is not None and sl_y1 is not None:
                    if check_bbox_intersect((x1, y1, x2, y2), top_roi_bbox) and cy <= sl_y1:
                        in_top_roi = True

                # 2. Xác định xe hiện tại có đang nằm trong ROI BOTTOM (polygon hoặc bounding box) hay không
                in_bottom_roi = False
                if bottom_roi is not None:
                    in_bottom_roi = is_in_roi((x1, y1, x2, y2), bottom_roi, method="bottom_center")
                elif sl_y2 is not None:
                    in_bottom_roi = (cy >= sl_y2)

                if not in_bottom_roi and bottom_roi_bbox is not None and sl_y2 is not None:
                    if check_bbox_intersect((x1, y1, x2, y2), bottom_roi_bbox) and cy >= sl_y2:
                        in_bottom_roi = True

                # 3. Xác định xe có đi vào hoặc chạm bounding box của STOPLINE hay không
                in_stopline_box = False
                if stopline_bbox is not None:
                    in_stopline_box = is_in_stopline_bbox(
                        vehicle_bbox=(x1, y1, x2, y2),
                        curr_pos=(cx, cy),
                        stopline_bbox=stopline_bbox,
                        padding_x=30,
                        padding_y=8,
                    ) or check_vehicle_crossed_stopline(
                        prev_pos=prev_pos,
                        curr_pos=(cx, cy),
                        vehicle_bbox=(x1, y1, x2, y2),
                        stopline_bbox=stopline_bbox,
                        padding_x=30,
                        padding_y=8,
                    )

                # Khởi tạo trạng thái cho xe mới xuất hiện
                if track_id not in self.vehicles:
                    if in_bottom_roi:
                        # Xe xuất hiện ở ROI BOTTOM -> Hợp lệ để theo dõi tiếp cận stopline
                        was_in_bottom = True
                        entered_stopline = False
                        is_recognized = True
                        init_status = VehicleStatus.APPROACHING
                        reason = "Tiếp cận vạch dừng (ROI Bottom)"
                    elif in_top_roi:
                        # Xe xuất hiện trực tiếp trong bounding box của ROI TOP (đi từ làn khác hoặc rẽ ngang vào):
                        # QUY TẮC: KHÔNG NHẬN DIỆN PHƯƠNG TIỆN ĐÓ!
                        was_in_bottom = False
                        entered_stopline = False
                        is_recognized = False
                        init_status = VehicleStatus.IGNORED
                        reason = "Xe từ làn khác đi vào ROI TOP (Không nhận diện)"
                    else:
                        was_in_bottom = False
                        entered_stopline = False
                        is_recognized = False
                        init_status = VehicleStatus.IGNORED
                        reason = "Ngoài vùng giám sát"

                    self.vehicles[track_id] = {
                        "status": init_status,
                        "class_name": vehicle_cls_name,
                        "start_frame": None,
                        "start_pos": None,
                        "lights_at_crossing": None,
                        "direction": "UNKNOWN",
                        "reason": reason,
                        "resolved_frame": None,
                        "bbox": (x1, y1, x2, y2),
                        "was_in_bottom": was_in_bottom,
                        "entered_stopline": entered_stopline,
                        "passed_stopline": False,
                        "is_recognized": is_recognized,
                        "last_frame": frame_idx,
                    }
                else:
                    self.vehicles[track_id]["bbox"] = (x1, y1, x2, y2)
                    self.vehicles[track_id]["class_name"] = vehicle_cls_name
                    self.vehicles[track_id]["last_frame"] = frame_idx

                v_info = self.vehicles[track_id]

                # Nếu xe đang trong ROI BOTTOM thì cập nhật was_in_bottom = True
                if in_bottom_roi:
                    v_info["was_in_bottom"] = True

                # A. KÍCH HOẠT BẮT ĐẦU THEO DÕI QUỸ ĐẠO:
                # "chỉ những xe ở ROI BOTTOM rồi đi qua bouding box của stopline thì bắt đầu theo dõi quỹ đạo xe trong ngưỡng thời gian là 3 giây"
                if v_info["was_in_bottom"] and not v_info.get("passed_stopline", False):
                    has_passed = False
                    if stopline_bbox is not None:
                        has_passed = check_vehicle_passed_stopline(
                            prev_pos=prev_pos,
                            curr_pos=(cx, cy),
                            vehicle_bbox=(x1, y1, x2, y2),
                            stopline_bbox=stopline_bbox,
                            padding_x=50,
                        )
                    if has_passed:
                        v_info["passed_stopline"] = True
                        v_info["entered_stopline"] = True
                        v_info["status"] = VehicleStatus.OBSERVING
                        v_info["start_frame"] = frame_idx  # Mốc bắt đầu tính 3 giây quan sát quỹ đạo
                        v_info["start_pos"] = (cx, cy)
                        v_info["lights_at_crossing"] = light_state.copy()
                        v_info["is_recognized"] = True
                        v_info["reason"] = f"Đi qua stopline -> Bắt đầu theo dõi ({self.observe_seconds}s)"
                        self.trajectories[track_id].append((cx, cy, frame_idx))

                # B. QUY TẮC NHẬN DIỆN PHƯƠNG TIỆN TRONG ROI TOP:
                # - Các phương tiện đi từ làn khác đi vào bounding box của ROI TOP sẽ KHÔNG nhận diện.
                # - CHỈ những xe ở ROI BOTTOM rồi đi qua bounding box của stopline thì mới nhận diện và theo dõi quỹ đạo.
                if in_top_roi:
                    if v_info["was_in_bottom"] and v_info.get("passed_stopline", False):
                        v_info["is_recognized"] = True
                        # Tiếp tục cập nhật quỹ đạo di chuyển trong ROI TOP
                        if v_info["status"] in (VehicleStatus.OBSERVING, VehicleStatus.VIOLATION, VehicleStatus.LEGITIMATE, VehicleStatus.LEGAL_PASS):
                            self.trajectories[track_id].append((cx, cy, frame_idx))
                    else:
                        # Xe từ làn khác đi vào ROI TOP -> Hoàn toàn KHÔNG NHẬN DIỆN!
                        v_info["is_recognized"] = False
                        v_info["status"] = VehicleStatus.IGNORED
                        v_info["reason"] = "Xe từ làn khác đi vào ROI TOP (Không nhận diện)"
                else:
                    # Xe ở ngoài ROI TOP (trong ROI BOTTOM hoặc đang dừng chờ trước vạch dừng):
                    if v_info["was_in_bottom"]:
                        v_info["is_recognized"] = True
                        if not v_info.get("passed_stopline", False):
                            v_info["status"] = VehicleStatus.APPROACHING
                    else:
                        v_info["is_recognized"] = False

                # 3. Xử lý xe đang trong thời gian QUAN SÁT QUỸ ĐẠO:
                # Phân định hướng di chuyển thông minh dựa trên độ sâu Y tiến vào giao lộ và tỷ lệ dạt ngang
                if v_info["status"] == VehicleStatus.OBSERVING and v_info.get("is_recognized", False):
                    start_frame = v_info["start_frame"]
                    start_x, start_y = v_info["start_pos"]
                    elapsed_frames = frame_idx - start_frame

                    dx = cx - start_x
                    dy = cy - start_y
                    forward_depth = -dy   # Xe di chuyển lên trên (cy < start_y) -> forward_depth dương

                    # Hướng trục đường tự nhiên theo phối cảnh camera:
                    # Khi tiến lên giao lộ (-dy), làn đường có góc nghiêng phối cảnh dịch sang trái ~ -0.80 * forward_depth.
                    # Độ dạt ngang thực tế so với trục làn đường:
                    lateral_offset = dx + 0.80 * forward_depth

                    # 1. Điều kiện phát hiện RẼ PHẢI:
                    # - Xe rẽ phải bắt buộc phải dạt sang bên phải (dx > 15px) và dạt mạnh so với trục đường (lateral_offset >= 40px)
                    # - Hoặc độ dạt sang phải cực lớn (dx >= 45px khi đã nhích được forward_depth >= 15px)
                    # - TUYỆT ĐỐI không bao giờ nhận diện rẽ phải nếu xe di chuyển lệch sang trái (dx <= 0)
                    is_turn_right = (
                        dx > 15 and forward_depth >= 15 and (lateral_offset >= 40 or dx >= 45)
                    )

                    # 2. Điều kiện phát hiện RẼ TRÁI:
                    # Phải rẽ ngoặt hẳn sang đường cắt ngang bên trái (vượt qua góc nghiêng tự nhiên của trục đường):
                    # - Lệch hẳn sang trái so với trục đường: lateral_offset <= -50px khi forward_depth >= 20px HOẶC
                    # - Xuất phát từ làn rẽ trái cạnh dải phân cách (start_x <= 450) và dạt mạnh sang trái (dx <= -50px)
                    is_turn_left = (forward_depth >= 20 and lateral_offset <= -50) or (
                        start_x <= 450 and dx <= -50 and forward_depth >= 15
                    )

                    # 3. Điều kiện kết luận ĐI THẲNG:
                    # - Xe bám sát trục đường tiến sâu vào ngã tư (forward_depth >= 30px và |lateral_offset| < 35px)
                    # - HOẶC xe đã tiến sâu qua ngưỡng deep_threshold_px mà không rẽ
                    # - HOẶC đã hết thời gian quan sát và đã tiến về phía trước >= 25px
                    is_straight_track = (forward_depth >= 30 and abs(lateral_offset) < 35)
                    is_straight_deep = (forward_depth >= self.deep_threshold_px and not is_turn_right and not is_turn_left)
                    is_straight_timeout = (elapsed_frames >= self.max_observe_frames and forward_depth >= 25 and not is_turn_right and not is_turn_left)

                    direction = None
                    if is_turn_right:
                        direction = "RIGHT"
                    elif is_turn_left:
                        direction = "LEFT"
                    elif is_straight_track or is_straight_deep or is_straight_timeout:
                        direction = "STRAIGHT"

                    if direction is not None:
                        v_info["direction"] = direction
                        v_info["resolved_frame"] = frame_idx

                        cross_lights = v_info["lights_at_crossing"] or light_state

                        # Đối chiếu hướng di chuyển với tín hiệu đèn tại thời điểm vượt vạch:
                        if direction == "RIGHT":
                            # Xe rẽ phải
                            if (
                                cross_lights.get("green_right_arrow", False)
                                or light_state.get("green_right_arrow", False)
                                or cross_lights.get("green_light", False)
                            ):
                                v_info["status"] = VehicleStatus.LEGITIMATE
                                v_info["reason"] = "Re phai hop le (co den re phai)"
                                if track_id not in self.legitimate_ids:
                                    self.legitimate_ids.add(track_id)
                                    self.legitimate_turn_count += 1
                                    self.legitimate_right_count += 1
                            else:
                                v_info["status"] = VehicleStatus.VIOLATION
                                v_info["reason"] = "Vi pham: Re phai khi den do"
                                if track_id not in self.violation_ids:
                                    self.violation_ids.add(track_id)
                                    self.violation_count += 1

                        elif direction == "STRAIGHT":
                            # Xe đi thẳng
                            if cross_lights.get("red_light", False) or cross_lights.get("red_left_arrow", False):
                                v_info["status"] = VehicleStatus.VIOLATION
                                v_info["reason"] = "Vi pham: Di thang khi den do"
                                if track_id not in self.violation_ids:
                                    self.violation_ids.add(track_id)
                                    self.violation_count += 1
                            else:
                                v_info["status"] = VehicleStatus.LEGAL_PASS
                                v_info["reason"] = "Di thang hop le (den xanh)"
                                self.straight_legal_count += 1

                        elif direction == "LEFT":
                            # Xe rẽ trái
                            if (
                                cross_lights.get("green_left_arrow", False)
                                or light_state.get("green_left_arrow", False)
                                or cross_lights.get("green_light", False)
                            ):
                                v_info["status"] = VehicleStatus.LEGITIMATE
                                v_info["reason"] = "Re trai hop le (co den re trai)"
                                if track_id not in self.legitimate_ids:
                                    self.legitimate_ids.add(track_id)
                                    self.legitimate_turn_count += 1
                                    self.legitimate_left_count += 1
                            else:
                                v_info["status"] = VehicleStatus.VIOLATION
                                v_info["reason"] = "Vi pham: Re trai khi den do"
                                if track_id not in self.violation_ids:
                                    self.violation_ids.add(track_id)
                                    self.violation_count += 1

        return light_state

    def draw_visuals(self, frame, frame_idx):
        """
        Vẽ toàn bộ hình ảnh trực quan:
        1. Vết quỹ đạo xe (Trajectory paths)
        2. Bounding box phân màu theo trạng thái (Xanh / Vàng theo dõi / Đỏ vi phạm / Xanh ngọc hợp lệ)
        3. Dashboard HUD thống kê trên cùng góc phải
        """
        h, w = frame.shape[:2]

        # 1. Vẽ các đường quỹ đạo di chuyển (chỉ vẽ xe được công nhận nhận diện)
        for track_id, info in self.vehicles.items():
            if not info.get("is_recognized", False):
                continue

            pts = list(self.trajectories[track_id])
            if len(pts) < 2:
                continue

            status = info["status"]
            if status == VehicleStatus.VIOLATION:
                path_color = (0, 0, 255)       # Đỏ: vi phạm
                thickness = 3
            elif status == VehicleStatus.LEGITIMATE:
                path_color = (0, 255, 128)     # Xanh ngọc: hợp lệ rẽ
                thickness = 2
            elif status == VehicleStatus.OBSERVING:
                path_color = (0, 215, 255)     # Vàng/Cam: đang theo dõi
                thickness = 2
            else:
                path_color = (200, 200, 200)   # Xám: bình thường
                thickness = 1

            for i in range(1, len(pts)):
                pt1 = (pts[i - 1][0], pts[i - 1][1])
                pt2 = (pts[i][0], pts[i][1])
                cv2.line(frame, pt1, pt2, path_color, thickness)

        # 2. Vẽ Bounding box và Nhãn trạng thái cho các xe đang xuất hiện (chỉ vẽ xe được công nhận nhận diện)
        for track_id, info in self.vehicles.items():
            if not info.get("is_recognized", False):
                continue

            # Chỉ vẽ nếu xe vừa xuất hiện gần đây (trong 2 frame)
            last_f = info.get("last_frame", 0)
            if frame_idx - last_f > 2:
                continue


            bbox = info.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            status = info["status"]
            cls_name = info.get("class_name", "xe")

            direction = info.get("direction", "UNKNOWN")
            # Map direction sang nhãn tiếng Việt
            dir_label = {"RIGHT": "Re phai", "LEFT": "Re trai", "STRAIGHT": "Di thang"}.get(direction, "")

            if status == VehicleStatus.VIOLATION:
                box_color = (0, 0, 255)
                dir_str = f" [{dir_label}]" if dir_label else ""
                tag = f"ID:{track_id} {cls_name}{dir_str} - VI PHAM VUOT DEN DO"
                bg_color = (0, 0, 200)
            elif status == VehicleStatus.LEGITIMATE:
                box_color = (0, 255, 128)
                dir_str = dir_label if dir_label else "Re"
                tag = f"ID:{track_id} {cls_name} - {dir_str} (HOP LE)"
                bg_color = (0, 160, 80)
            elif status == VehicleStatus.LEGAL_PASS:
                box_color = (0, 220, 100)
                dir_str = dir_label if dir_label else "Di thang"
                tag = f"ID:{track_id} {cls_name} - {dir_str} (HOP LE)"
                bg_color = (0, 140, 60)
            elif status == VehicleStatus.OBSERVING:
                box_color = (0, 215, 255)
                elapsed = frame_idx - (info["start_frame"] or frame_idx)
                time_left = max(0.0, (self.max_observe_frames - elapsed) / self.fps)
                dir_str = f" [{dir_label}]" if dir_label else ""
                tag = f"ID:{track_id} {cls_name}{dir_str} - Theo doi ({time_left:.1f}s)"
                bg_color = (0, 140, 200)
            else:
                box_color = (0, 255, 0)
                tag = f"ID:{track_id} {cls_name}"
                bg_color = (0, 180, 0)

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Vẽ nhãn với nền
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 8, th + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (min(w - 1, x1 + tw + 6), label_y + 4),
                bg_color,
                -1,
            )
            cv2.putText(
                frame,
                tag,
                (x1 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Vẽ Dashboard HUD thông tin ở góc trên bên phải
        panel_w, panel_h = 360, 125
        px1 = w - panel_w - 15
        py1 = 15
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        # Nền bán trong suốt cho HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)

        # Tiêu đề HUD
        cv2.putText(
            frame,
            "HE THONG PHAN TICH QUY DAO & VI PHAM",
            (px1 + 10, py1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Số xe đang theo dõi quỹ đạo (chỉ tính các xe đang xuất hiện gần đây)
        num_observing = sum(1 for v in self.vehicles.values() if v["status"] == VehicleStatus.OBSERVING and v.get("is_recognized", False) and (frame_idx - v.get("last_frame", 0) <= 5))
        cv2.putText(
            frame,
            f"Dang quan sat quy dao: {num_observing} xe (Nguong {self.observe_seconds}s)",
            (px1 + 10, py1 + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Số xe vi phạm
        cv2.putText(
            frame,
            f"Vi pham vuot den do: {self.violation_count} xe",
            (px1 + 10, py1 + 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Số xe rẽ hợp lệ (phải + trái)
        cv2.putText(
            frame,
            f"Re hop le: {self.legitimate_turn_count} (P:{self.legitimate_right_count} T:{self.legitimate_left_count})",
            (px1 + 10, py1 + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

        return frame
