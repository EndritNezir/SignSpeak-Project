from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Model Load کرو
model = load_model('gesture_recognition_model.h5')

# Gesture Dictionary (Manually define کرو)
gesture_dict = {i: chr(65 + i) for i in range(26)}  # A-Z gestures

def predict_gesture(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    return gesture_dict.get(predicted_class, "Unknown Gesture")

# Test کرو
test_image = "path_to_test_image.jpg"
result = predict_gesture(test_image)
print(f'Predicted Gesture: {result}')
