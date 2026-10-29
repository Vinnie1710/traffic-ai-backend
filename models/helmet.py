from ultralytics import YOLO 
model = YOLO("weights/helmet_26s.pt") 
def detect_helmet(image):
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose = False 
    )
    return result 