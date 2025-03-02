import cv2
import os

GESTURE_NAME = "Hello"  # Change this for different gestures
SAVE_PATH = f"dataset/{GESTURE_NAME}/"

os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collecting Images - Press 'c' to capture, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        img_name = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
