from ultralytics import YOLO 
model = YOLO("weights/light_traffic-26m-best.pt")

def detect_traffic_light(image):
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose = False 

    )
    return result 