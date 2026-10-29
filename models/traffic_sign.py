from ultralytics import YOLO 
model = YOLO("weights/OCR-26s-best.pt")

def detect_traffic_sign(image):
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose = False

    )
    return result