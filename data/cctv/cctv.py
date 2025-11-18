import cv2
import time
import threading
import numpy as np
from playsound import playsound

# ----------------------------
# Human Detection (HOG)
# ----------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ----------------------------
# Video Capture (Laptop / Mobile)
# ----------------------------
# Laptop camera (DirectShow backend for Windows fix)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 👉 Use this line instead if you want mobile camera (replace IP with your phone IP from IP Webcam app)
# cap = cv2.VideoCapture("http://192.168.1.5:8080/video")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ----------------------------
# Alert Settings
# ----------------------------
last_alert_time = 0
alert_cooldown = 5  # seconds

def play_alert_sound():
    try:
        playsound("alert.mp3")
    except:
        print("⚠️ Could not play alert sound")

# ----------------------------
# Motion Detection
# ----------------------------
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
min_motion_area = 500  # Minimum motion area

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not grab frame. Check camera connection.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Motion detection
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = any(cv2.contourArea(c) > min_motion_area for c in contours)

    # Human detection only if motion detected
    humans, weights = [], []
    if motion_detected:
        humans, weights = hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        # Debug info
        print(f"Detections: {len(humans)}, Weights: {weights}")

    # Draw detections + trigger alert
    for i, (x, y, w, h) in enumerate(humans):
        if weights[i] > 0.2:  # Lower threshold
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Human {weights[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                threading.Thread(target=play_alert_sound).start()
                last_alert_time = current_time

    # Display status
    status = f"Humans: {len(humans)} | Motion: {'Yes' if motion_detected else 'No'}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Enhanced Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
