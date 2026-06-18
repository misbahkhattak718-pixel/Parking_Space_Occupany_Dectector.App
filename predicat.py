from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model("test.jpg", save=True)

print("Prediction Done")