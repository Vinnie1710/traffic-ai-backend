from ultralytics import YOLO 
model = YOLO("weights/LP-26s-best.pt") 
def detect_license_plate(image):
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose = False 
    )
    return result 