import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 1. LOAD DATASET
def load_frames(video_path, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return np.array(frames)

frames = load_frames("traffic.mp4")

# 2. NUMBER OF FRAMES & FEATURES
print(f"Total Frames : {len(frames)}")          # 864
print(f"Frame Shape  : {frames.shape[1:]}")     # (224, 224, 3)
print(f"Full Shape   : {frames.shape}")         # (864, 224, 224, 3)
print(f"Feature Dim  : {np.prod(frames.shape[1:])}")  # 150,528 per frame

# 3. TRAIN / TEST SPLIT
labels = np.zeros(len(frames), dtype=np.int64)  # replace with real labels

X_train, X_test, y_train, y_test = train_test_split(
    frames, labels, test_size=0.2, random_state=42
)

print(f"Train : {X_train.shape}")   # (691, 224, 224, 3)
print(f"Test  : {X_test.shape}")    # (173, 224, 224, 3)

# 4. DATA FORMAT — normalize to [0, 1]
X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32)  / 255.0

print(f"dtype : {X_train.dtype}")
print(f"range : [{X_train.min():.1f}, {X_train.max():.1f}]")