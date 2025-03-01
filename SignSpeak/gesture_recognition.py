import cv2
import numpy as np
import tensorflow as tf

class GestureRecognition:
    def __init__(self, model_path="models/gesture_model.h5", labels=None):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = labels if labels else ["Hello", "Yes", "No", "Thank You", "Please"]

    def preprocess_image(self, hand_img):
        hand_img = cv2.resize(hand_img, (64, 64))  # Resize for model
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        hand_img = hand_img / 255.0  # Normalize
        hand_img = np.expand_dims(hand_img, axis=[0, -1])  # Add batch & channel dims
        return hand_img

    def predict_gesture(self, hand_img):
        processed_img = self.preprocess_image(hand_img)
        prediction = self.model.predict(processed_img)
        return self.labels[np.argmax(prediction)]

if __name__ == "__main__":
    # Test Gesture Recognition with a Sample Image
    recognizer = GestureRecognition()
    
    test_img = cv2.imread("dataset/test_image.jpg")  # Replace with an actual test image
    predicted_gesture = recognizer.predict_gesture(test_img)

    print(f"Predicted Gesture: {predicted_gesture}")
