import cv2
from ultralytics import YOLO

# ----------------------------
# 1. Load YOLOv8 model
# ----------------------------
model = YOLO("yolov8n.pt")  # small YOLOv8 model

# ----------------------------
# 2. Video Input & Output
# ----------------------------
video_path = "parking_lot_video (2).mp4"  # replace with your video file
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output_detected.mp4', fourcc, 20.0, (frame_width, frame_height))

# ----------------------------
# 3. Detection Settings
# ----------------------------
ALLOWED_CLASSES = [2, 3]  # 2=car, 3=motorbike
CONFIDENCE_THRESHOLD = 0.3  # reduce if some cars are missed

# ----------------------------
# 4. Process Video
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    for i, box in enumerate(boxes):
        class_id = int(labels[i])
        if class_id not in ALLOWED_CLASSES:
            continue  # skip non-car/bike

        x1, y1, x2, y2 = box
        vehicle_type = "Car" if class_id == 2 else "Bike"

        # Draw bounding box
        color = (0, 255, 0) if vehicle_type == "Car" else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, vehicle_type, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save output frame
    out.write(frame)
    cv2.imshow("Parking Lot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Detection complete. Output saved as output_detected.mp4")
