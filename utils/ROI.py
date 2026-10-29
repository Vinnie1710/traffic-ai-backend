# Module quản lý, tạo và vẽ các vùng ROI (Region of Interest) hình thang động dựa trên bounding box của stopline
import cv2
import numpy as np


def get_stopline_rois(
    stopline_bbox,
    frame_shape=None,
    expand_top=0.22,
    expand_bottom=0.25,
    padding_x=0,
    full_width_top=True,
    full_width_bottom=True,
):
    """
    Tính toán tọa độ 2 vùng ROI dạng hình thang dựa trên stopline:
    - ROI trên: hình thang ngược, đáy bé tại cạnh trên của stopline,
      cạnh trên ở đỉnh khung hình (y = 0) trải dài toàn bộ chiều ngang khung hình (full_width_top=True).
    - ROI dưới: hình thang xuôi, đáy bé tại cạnh dưới của stopline,
      cạnh dưới ở đáy khung hình (y = height - 1) trải dài toàn bộ chiều ngang khung hình (full_width_bottom=True).

    Args:
        stopline_bbox (tuple/list): Tọa độ (x1, y1, x2, y2) của stopline.
        frame_shape (tuple, optional): (height, width) để clamp tọa độ trong khung hình.
        expand_top (float/int): Độ mở rộng sang 2 bên ở đỉnh khung hình (khi full_width_top=False).
                                Nếu là float (0 < x < 1): tính theo tỉ lệ chiều rộng stopline (mặc định 0.22).
                                Nếu là int: số pixel mở rộng cố định.
        expand_bottom (float/int): Độ mở rộng sang 2 bên ở đáy khung hình (khi full_width_bottom=False).
                                   Nếu là float (0 < x < 1): tính theo tỉ lệ chiều rộng stopline (mặc định 0.25).
                                   Nếu là int: số pixel mở rộng cố định.
        padding_x (int, optional): Độ mở rộng chiều ngang của ROI tại vị trí stopline.
        full_width_top (bool, optional): Nếu True (mặc định), cạnh trên của ROI Top sẽ bằng toàn bộ chiều ngang khung hình.
        full_width_bottom (bool, optional): Nếu True (mặc định), cạnh dưới của ROI Bottom sẽ bằng toàn bộ chiều ngang khung hình.

    Returns:
        tuple: (top_roi, bottom_roi) với mỗi ROI là numpy array shape (4, 2) biểu diễn 4 đỉnh của hình thang.
    """
    if stopline_bbox is None:
        return None, None

    x1, y1, x2, y2 = map(int, stopline_bbox)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    stopline_w = max(x_max - x_min, 20)

    # Tính pixel mở rộng ở đỉnh khung hình (ROI trên) nếu không dùng full width
    if isinstance(expand_top, float) and 0 < expand_top < 1.0:
        exp_top = int(stopline_w * expand_top)
    else:
        exp_top = int(expand_top) if expand_top is not None else int(stopline_w * 0.22)

    # Tính pixel mở rộng ở đáy khung hình (ROI dưới) nếu không dùng full width
    if isinstance(expand_bottom, float) and 0 < expand_bottom < 1.0:
        exp_bottom = int(stopline_w * expand_bottom)
    else:
        exp_bottom = int(expand_bottom) if expand_bottom is not None else int(stopline_w * 0.25)

    # Chiều rộng tại vị trí stopline
    sl_x1 = x_min - padding_x
    sl_x2 = x_max + padding_x

    if frame_shape is not None:
        h, w = frame_shape[:2]
        sl_x1 = max(0, min(sl_x1, w - 1))
        sl_x2 = max(0, min(sl_x2, w - 1))

        # Tọa độ cạnh trên của ROI trên ở đỉnh khung hình (y = 0)
        if full_width_top:
            top_x1 = 0
            top_x2 = w - 1
        else:
            top_x1 = max(0, min(sl_x1 - exp_top, w - 1))
            top_x2 = max(0, min(sl_x2 + exp_top, w - 1))

        # Tọa độ cạnh dưới của ROI dưới ở đáy khung hình (y = h - 1)
        if full_width_bottom:
            bottom_x1 = 0
            bottom_x2 = w - 1
        else:
            bottom_x1 = max(0, min(sl_x1 - exp_bottom, w - 1))
            bottom_x2 = max(0, min(sl_x2 + exp_bottom, w - 1))

        # ROI trên (hình thang ngược):
        # Đỉnh khung hình (y=0) bằng toàn bộ chiều ngang: [top_x1, 0] -> [top_x2, 0]
        # Chạm stopline (y=y_min): [sl_x2, y_min] -> [sl_x1, y_min]
        top_roi = np.array([
            [top_x1, 0],
            [top_x2, 0],
            [sl_x2, max(0, min(y_min, h - 1))],
            [sl_x1, max(0, min(y_min, h - 1))],
        ], dtype=np.int32)

        # ROI dưới (hình thang xuôi):
        # Chạm stopline (y=y_max): [sl_x1, y_max] -> [sl_x2, y_max]
        # Đáy khung hình (y=h-1) bằng toàn bộ chiều ngang: [bottom_x2, h - 1] -> [bottom_x1, h - 1]
        bottom_roi = np.array([
            [sl_x1, max(0, min(y_max, h - 1))],
            [sl_x2, max(0, min(y_max, h - 1))],
            [bottom_x2, h - 1],
            [bottom_x1, h - 1],
        ], dtype=np.int32)
    else:
        top_x1 = sl_x1 - exp_top
        top_x2 = sl_x2 + exp_top
        bottom_x1 = sl_x1 - exp_bottom
        bottom_x2 = sl_x2 + exp_bottom

        top_roi = np.array([
            [top_x1, 0],
            [top_x2, 0],
            [sl_x2, y_min],
            [sl_x1, y_min],
        ], dtype=np.int32)

        bottom_roi = np.array([
            [sl_x1, y_max],
            [sl_x2, y_max],
            [bottom_x2, y_max + 300],
            [bottom_x1, y_max + 300],
        ], dtype=np.int32)

    return top_roi, bottom_roi


def draw_roi(
    frame,
    roi,
    label="",
    color=(255, 0, 0),
    thickness=2,
    fill_alpha=0.15,
):
    """
    Hàm vẽ 1 vùng ROI (hình thang polygon hoặc hình chữ nhật) lên frame với viền, nền bán trong suốt và nhãn.

    Args:
        frame (np.ndarray): Frame ảnh hiện tại.
        roi: Numpy array các đỉnh polygon (N, 2) hoặc bounding box (x1, y1, x2, y2).
        label (str): Nhãn hiển thị cho vùng ROI.
        color (tuple): Mã màu BGR.
        thickness (int): Độ dày đường viền.
        fill_alpha (float): Độ trong suốt nền (0.0 đến 1.0).

    Returns:
        np.ndarray: Frame ảnh đã được vẽ ROI.
    """
    if roi is None or frame is None:
        return frame

    # Hỗ trợ cả polygon (N, 2) và bounding box (x1, y1, x2, y2)
    if isinstance(roi, np.ndarray) or (isinstance(roi, (list, tuple)) and len(roi) > 0 and hasattr(roi[0], "__len__")):
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
    else:
        x1, y1, x2, y2 = map(int, roi)
        pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)

    # Đổ màu bán trong suốt cho vùng ROI
    if fill_alpha > 0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    # Vẽ nét viền cho vùng ROI
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    # Hiển thị nhãn tên vùng ROI
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Lấy tọa độ điểm đầu tiên (góc trên bên trái của ROI)
        lx = int(pts[0, 0, 0])
        ly = int(pts[0, 0, 1])

        bg_y1 = ly
        bg_y2 = min(frame.shape[0] - 1, ly + th + 8)
        text_y = min(frame.shape[0] - 1, ly + th + 4)

        cv2.rectangle(frame, (lx, bg_y1), (min(frame.shape[1] - 1, lx + tw + 8), bg_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (lx + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


# Alias để đảm bảo tương thích ngược
draw_roi_box = draw_roi


def draw_stopline_rois(
    frame,
    stopline_bbox,
    expand_top=0.22,
    expand_bottom=0.25,
    padding_x=0,
    full_width_top=True,
    full_width_bottom=True,
    top_label="ROI Top",
    bottom_label="ROI Bottom",
    color_top=(255, 191, 0),       # Xanh dương sáng (BGR)
    color_bottom=(0, 165, 255),    # Màu cam (BGR)
    thickness=2,
    fill_alpha=0.18,
):
    """
    Hàm vẽ 2 vùng ROI hình thang từ bounding box của stopline:
    - ROI trên: hình thang ngược (cạnh trên ở y=0 bằng toàn bộ chiều ngang khung hình khi full_width_top=True)
    - ROI dưới: hình thang xuôi (cạnh dưới ở y=height-1 bằng toàn bộ chiều ngang khung hình khi full_width_bottom=True)
    Mỗi video đưa vào sẽ tự động tạo ra các ROI tương ứng với video đó.

    Args:
        frame (np.ndarray): Frame ảnh hiện tại của video.
        stopline_bbox (tuple/list): Bounding box của stopline (x1, y1, x2, y2).
        expand_top (float/int): Độ mở rộng ở đỉnh khung hình (khi full_width_top=False).
        expand_bottom (float/int): Độ mở rộng ở đáy khung hình (khi full_width_bottom=False).
        padding_x (int, optional): Mở rộng lề chiều ngang tại vị trí stopline.
        full_width_top (bool, optional): Cạnh trên ROI Top trải dài toàn bộ chiều ngang khung hình (mặc định True).
        full_width_bottom (bool, optional): Cạnh dưới ROI Bottom trải dài toàn bộ chiều ngang khung hình (mặc định True).
        top_label (str): Nhãn hiển thị cho ROI phía trên.
        bottom_label (str): Nhãn hiển thị cho ROI phía dưới.
        color_top (tuple): Mã màu BGR cho ROI trên.
        color_bottom (tuple): Mã màu BGR cho ROI dưới.
        thickness (int): Độ dày viền vẽ.
        fill_alpha (float): Độ mờ trong suốt của nền ROI (0.0 đến 1.0).

    Returns:
        tuple: (top_roi, bottom_roi)
    """
    if stopline_bbox is None or frame is None:
        return None, None

    h, w = frame.shape[:2]
    top_roi, bottom_roi = get_stopline_rois(
        stopline_bbox,
        frame_shape=(h, w),
        expand_top=expand_top,
        expand_bottom=expand_bottom,
        padding_x=padding_x,
        full_width_top=full_width_top,
        full_width_bottom=full_width_bottom,
    )

    # Vẽ ROI trên (hình thang ngược)
    draw_roi(
        frame,
        top_roi,
        label=top_label,
        color=color_top,
        thickness=thickness,
        fill_alpha=fill_alpha,
    )

    # Vẽ ROI dưới (hình thang xuôi)
    draw_roi(
        frame,
        bottom_roi,
        label=bottom_label,
        color=color_bottom,
        thickness=thickness,
        fill_alpha=fill_alpha,
    )

    return top_roi, bottom_roi


def is_point_in_roi(point, roi):
    """
    Hàm tiện ích kiểm tra xem một điểm (x, y) có nằm trong vùng ROI (hình thang hoặc chữ nhật) hay không.
    Dùng cho logic phát hiện phương tiện đi vào ROI / vượt đèn đỏ sau này.

    Args:
        point (tuple/list): Tọa độ (x, y) của điểm cần kiểm tra.
        roi: Numpy array polygon (4, 2) hoặc bounding box (x1, y1, x2, y2).

    Returns:
        bool: True nếu điểm nằm trong ROI, False nếu nằm ngoài.
    """
    if roi is None or point is None:
        return False

    if isinstance(roi, np.ndarray) or (isinstance(roi, (list, tuple)) and len(roi) > 0 and hasattr(roi[0], "__len__")):
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
    else:
        x1, y1, x2, y2 = map(int, roi)
        pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)

    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def check_bbox_intersect(box1, box2, padding=0):
    """
    Kiểm tra 2 bounding box (x1, y1, x2, y2) có giao nhau (overlap) hay không.
    Hỗ trợ mở rộng lề padding cho box2.
    """
    if box1 is None or box2 is None:
        return False
    a_x1, a_y1, a_x2, a_y2 = map(int, box1)
    b_x1, b_y1, b_x2, b_y2 = map(int, box2)

    b_x1 -= padding
    b_x2 += padding
    b_y1 -= padding
    b_y2 += padding

    # Điều kiện không giao nhau
    if a_x2 < b_x1 or a_x1 > b_x2 or a_y2 < b_y1 or a_y1 > b_y2:
        return False
    return True


def get_roi_bbox(roi):
    """
    Lấy bounding box (x_min, y_min, x_max, y_max) từ ROI (polygon hoặc bbox).
    Dùng để kiểm tra giao cắt hoặc đối tượng nằm trong bounding box của ROI.
    """
    if roi is None:
        return None
    if isinstance(roi, np.ndarray) or (isinstance(roi, (list, tuple)) and len(roi) > 0 and hasattr(roi[0], "__len__")):
        pts = np.array(roi, dtype=np.int32).reshape((-1, 2))
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        return (x_min, y_min, x_max, y_max)
    x1, y1, x2, y2 = map(int, roi)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def is_in_roi(vehicle_bbox, roi, method="bottom_center", check_bbox=True):
    """
    Kiểm tra xem xe có nằm trong vùng ROI (hình thang hoặc bounding box của ROI) hay không:
    - Kiểm tra điểm đại diện (bottom_center hoặc center) nằm trong polygon ROI.
    - Hoặc kiểm tra bounding box của xe có giao cắt/nằm trong bounding box của ROI.

    Args:
        vehicle_bbox (tuple): (x1, y1, x2, y2) của xe.
        roi: Polygon (N, 2) hoặc Bounding box (x1, y1, x2, y2) của ROI.
        method (str): 'bottom_center' (mặc định) hoặc 'center'.
        check_bbox (bool): Nếu True, kiểm tra thêm tương tác với bounding box của ROI.

    Returns:
        bool: True nếu xe nằm trong ROI hoặc bounding box của ROI.
    """
    if vehicle_bbox is None or roi is None:
        return False

    x1, y1, x2, y2 = map(int, vehicle_bbox)
    if method == "center":
        pt = ((x1 + x2) // 2, (y1 + y2) // 2)
    else:
        pt = ((x1 + x2) // 2, y2)

    # 1. Kiểm tra điểm đại diện có nằm trong polygon ROI không
    if is_point_in_roi(pt, roi):
        return True

    # 2. Kiểm tra giao cắt với bounding box của ROI
    if check_bbox:
        roi_bbox = get_roi_bbox(roi)
        if roi_bbox is not None:
            r_x1, r_y1, r_x2, r_y2 = roi_bbox
            # Xe giao cắt bbox và tâm đáy nằm trong khoảng Y của ROI
            if check_bbox_intersect(vehicle_bbox, roi_bbox):
                if r_y1 <= pt[1] <= r_y2:
                    return True

    return False


def is_bbox_in_roi(bbox, roi, method="bottom_center"):
    """
    Kiểm tra xem bounding box của xe có nằm trong ROI hay không theo điểm đại diện:
    - 'bottom_center': tâm đáy xe (tiếp xúc mặt đường) ((x1+x2)//2, y2)
    - 'center': tâm xe ((x1+x2)//2, (y1+y2)//2)
    """
    return is_in_roi(bbox, roi, method=method, check_bbox=False)


def is_in_stopline_bbox(
    vehicle_bbox,
    curr_pos=None,
    stopline_bbox=None,
    padding_x=25,
    padding_y=8,
):
    """
    Kiểm tra xem xe có đi vào hoặc chạm vào bounding box của stopline hay không.
    Dùng để kích hoạt mốc bắt đầu theo dõi quỹ đạo xe trong 3 giây khi xe từ ROI BOTTOM đi vào stopline.

    Args:
        vehicle_bbox (tuple): (x1, y1, x2, y2) của xe.
        curr_pos (tuple, optional): (cx, cy) tâm đáy xe.
        stopline_bbox (tuple): (sx1, sy1, sx2, sy2) bounding box của stopline.
        padding_x (int): Độ mở rộng chiều ngang stopline.
        padding_y (int): Độ mở rộng chiều dọc stopline.

    Returns:
        bool: True nếu xe đã đi vào hoặc giao cắt với bounding box của stopline.
    """
    if stopline_bbox is None:
        return False

    sx1, sy1, sx2, sy2 = map(int, stopline_bbox)
    sl_x1, sl_x2 = min(sx1, sx2) - padding_x, max(sx1, sx2) + padding_x
    sl_y1, sl_y2 = min(sy1, sy2) - padding_y, max(sy1, sy2) + padding_y

    # 1. Bounding box của xe giao cắt với bounding box stopline
    if vehicle_bbox is not None:
        if check_bbox_intersect(vehicle_bbox, (sl_x1, sl_y1, sl_x2, sl_y2)):
            return True

    # 2. Tâm đáy xe nằm trong vùng stopline
    if curr_pos is not None:
        cx, cy = curr_pos
        if sl_x1 <= cx <= sl_x2 and sl_y1 <= cy <= sl_y2:
            return True

    return False


def check_vehicle_crossed_stopline(
    prev_pos,
    curr_pos,
    vehicle_bbox,
    stopline_bbox,
    padding_x=30,
    padding_y=8,
):
    """
    Kiểm tra xem xe có đi qua hoặc chạm vào bounding box của stopline theo hướng tiến vào ROI TOP hay không.

    Args:
        prev_pos (tuple): (x, y) vị trí tâm đáy xe ở frame trước (nếu có).
        curr_pos (tuple): (x, y) vị trí tâm đáy xe ở frame hiện tại.
        vehicle_bbox (tuple): (vx1, vy1, vx2, vy2) của xe.
        stopline_bbox (tuple): (sx1, sy1, sx2, sy2) của stopline.
        padding_x (int): Độ mở rộng chiều ngang stopline.
        padding_y (int): Độ mở rộng chiều dọc stopline.

    Returns:
        bool: True nếu xe đi qua / giao cắt với stopline theo chiều tiến vào ROI TOP.
    """
    if stopline_bbox is None or curr_pos is None:
        return False

    sx1, sy1, sx2, sy2 = map(int, stopline_bbox)
    sl_x1, sl_x2 = min(sx1, sx2) - padding_x, max(sx1, sx2) + padding_x
    sl_y1, sl_y2 = min(sy1, sy2) - padding_y, max(sy1, sy2) + padding_y

    cx, cy = curr_pos

    # 1. Bounding box của xe giao cắt trực tiếp với bounding box của stopline
    if vehicle_bbox is not None:
        if check_bbox_intersect(vehicle_bbox, (sl_x1, sl_y1, sl_x2, sl_y2)):
            return True

    # 2. Tâm đáy xe nằm trong vùng stopline
    if sl_x1 <= cx <= sl_x2 and sl_y1 <= cy <= sl_y2:
        return True

    # 3. Quỹ đạo giữa 2 frame liên tiếp cắt qua đường stopline (theo hướng tiến lên y giảm dần)
    if prev_pos is not None:
        px, py = prev_pos
        if py >= sl_y1 and cy <= sl_y2 and sl_x1 <= cx <= sl_x2:
            return True

    return False


def check_vehicle_passed_stopline(
    prev_pos,
    curr_pos,
    vehicle_bbox,
    stopline_bbox,
    padding_x=50,
):
    """
    Kiểm tra xem xe đã thực sự ĐI QUA bounding box của stopline tiến vào ROI TOP (giao lộ) hay chưa:
    - Xe đến từ làn đường phía dưới (ROI BOTTOM).
    - Tâm đáy xe cy đã vượt qua mép trên của stopline (cy <= sl_y1 + 5).
    - Hoặc tâm xe mid_y đã vượt qua mép trên của stopline (mid_y < sl_y1) và xe đang di chuyển tiến về phía trước.
    - Tuyệt đối KHÔNG kích hoạt đối với các xe dừng chờ đèn đỏ trước hoặc chạm mép stopline.

    Args:
        prev_pos (tuple): (x, y) vị trí frame trước.
        curr_pos (tuple): (cx, cy) tâm đáy xe (tiếp xúc mặt đường).
        vehicle_bbox (tuple): (vx1, vy1, vx2, vy2) của xe.
        stopline_bbox (tuple): (sx1, sy1, sx2, sy2) của stopline.
        padding_x (int): Độ mở rộng chiều ngang của stopline.

    Returns:
        bool: True nếu xe đã đi qua vạch dừng tiến vào giao lộ.
    """
    if stopline_bbox is None or curr_pos is None or vehicle_bbox is None:
        return False

    sx1, sy1, sx2, sy2 = map(int, stopline_bbox)
    sl_x1, sl_x2 = min(sx1, sx2) - padding_x, max(sx1, sx2) + padding_x
    sl_y1 = min(sy1, sy2)  # Mép trên của stopline (hướng về ngã tư / ROI TOP)
    sl_y2 = max(sy1, sy2)  # Mép dưới của stopline (hướng về ROI BOTTOM)

    vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)
    cx, cy = curr_pos
    mid_y = (vy1 + vy2) // 2

    # Xe phải nằm trong phạm vi chiều ngang của stopline (hoặc bbox xe giao với dải stopline)
    if not (sl_x1 <= cx <= sl_x2 or (vx1 <= sl_x2 and vx2 >= sl_x1)):
        return False

    # 1. Tâm đáy xe (bánh xe sau / điểm tiếp xúc mặt đường) đã vượt qua mép trên stopline:
    # Xe đã hoàn toàn đi qua vạch dừng tiến vào giao lộ
    if cy <= sl_y1 + 5:
        return True

    # 2. Tâm xe đã qua mép trên của stopline và xe đang di chuyển tiến lên (cy giảm dần):
    # Dành cho xe thân dài (xe tải, ô tô) khi phần lớn thân xe đã qua vạch dừng
    if mid_y < sl_y1 and cy < sl_y2 + 10:
        if prev_pos is not None:
            py = prev_pos[1]
            if py - cy >= 2:  # Đang di chuyển tiến lên phía trước ít nhất 2px
                return True
        elif vy1 < sl_y1 - 30:  # Đầu xe đã tiến sâu vào giao lộ ít nhất 30px
            return True

    # 3. Quỹ đạo giữa 2 frame liên tiếp cắt qua mép trên stopline (sl_y1)
    if prev_pos is not None:
        px, py = prev_pos
        if py >= sl_y1 and cy < sl_y1:
            return True

    return False

