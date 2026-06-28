import cv2
from ultralytics import YOLO

# ── Load YOLOv8 model ──
model = YOLO('yolov8n.pt')  # downloads automatically first time
print("YOLOv8 Model loaded successfully!")

# ── Load Video ──
cap = cv2.VideoCapture('traffic.mp4')

if not cap.isOpened():
    print(" ERROR: Video file not found.")
else:
    print(" Video loaded successfully!")

# ── Get video info ──
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
duration     = total_frames / fps

print(f" Total Frames : {total_frames}")
print(f"FPS          : {fps}")
print(f" Duration     : {duration:.2f} seconds")
print("─" * 40)

# ── Vehicle classes in COCO dataset ──
VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck', 1: 'Bicycle'}

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp    = frame_id / fps
    frame_resized = cv2.resize(frame, (640, 640))

    # ── Run YOLOv8 detection ──
    results = model(frame_resized, verbose=False)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])

            # ── Only process vehicle classes ──
            if class_id in VEHICLE_CLASSES and confidence > 0.5:
                vehicle_count += 1

    # ── Display only the vehicle count on the frame ──
    cv2.putText(frame_resized, f"Vehicles: {vehicle_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # ── Show video window ──
    cv2.imshow("Vehicle Count - YOLOv8", frame_resized)

    # ── Print to terminal ──
    print(f"Frame {frame_id} / {total_frames} | Time: {timestamp:.2f}s | Vehicles: {vehicle_count}")

    # ── Press Q to quit ──
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

print("─" * 40)
print(f"Total Frames Counted  : {frame_id}")
print(f" Total Frames in Video : {total_frames}")
print(f"  Total Duration        : {duration:.2f} seconds")
print("Done.")