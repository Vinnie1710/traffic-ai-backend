from ultralytics import YOLO
model = YOLO("weights/stop_line-26s-best.pt")
def detect_stopline(image):
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose = False
    )
    return result