import cv2
from yolov5 import YOLOv5
import winsound  # Import winsound for beep functionality

# Load YOLOv5 model
model = YOLOv5("yolov5s.pt")  # Pretrained YOLOv5 model

# Define harmful objects
harmful_objects = ["knife", "gun", "scissors"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame)

    # Parse results
    harmful_detected = False

    for detection in results.xyxy[0]:  # Iterate over detections
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
        conf, cls = detection[4:]  # Confidence and class index
        class_name = results.names[int(cls)]

        # Check if detected object is harmful
        if class_name in harmful_objects:
            harmful_detected = True
            winsound.Beep(1000, 500)  # Beep sound when harmful object is detected
            color = (0, 0, 255)  # Red for harmful objects
        else:
            color = (0, 255, 0)  # Green for safe objects

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Display overall label
    label = "Harmful" if harmful_detected else "Not Harmful"
    label_color = (0, 0, 255) if harmful_detected else (0, 255, 0)
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 3)

    # Show frame
    cv2.imshow("Object Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
