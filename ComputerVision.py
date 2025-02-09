import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load model

# Replace with your iPhone's streaming URL
url = "http://192.168.1.100:8080/video"
cap = cv2.VideoCapture(url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Safety Gear Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
