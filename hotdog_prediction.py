from ultralytics import YOLO
import cv2

# Load your trained model (.pt file)
model = YOLO("weights.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction (stream=True returns generator of results)
    results = model.predict(frame, imgsz=960, conf=0.30)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLO Detection", annotated_frame)

    # Exit when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
