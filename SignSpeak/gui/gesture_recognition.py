import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Mapping
gesture_dict = {
    "thumbs_up": "Yes",
    "fist": "Stop",
    "open_palm": "Hello"
    
}

def recognize_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Simple Gesture Detection (Add ML model for better accuracy)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if thumb_tip.y < index_tip.y:  # Thumbs Up
                return "thumbs_up"
            elif all(lm.y > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y for lm in hand_landmarks.landmark):
                return "open_palm"
            else:
                return "fist"
    
    return None
