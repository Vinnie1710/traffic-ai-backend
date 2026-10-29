from ultralytics import YOLO 
model = YOLO("weights/vehicle-26s-best.pt")
def detect_vehicle(image, track=True):
    if track:
        result = model.track(
            source=image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False,
        )
    else:
        result = model.predict(
            source=image,
            conf=0.25,
            verbose=False,
        )
    return result


def track_vehicle(image):
    return detect_vehicle(image, track=True)