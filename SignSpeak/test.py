from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.keras')

gesture_dict = {i: str(i) for i in range(10)}  # 0-9
gesture_dict.update({i + 10: chr(65 + i) for i in range(26)})  # A-Z


def predict_gesture(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    return gesture_dict.get(predicted_class, "Unknown Gesture")

test_image = "\Fiverr Projects\SignSpeak system\SignSpeak\hand1_3_bot_seg_3_cropped.jpeg"
result = predict_gesture(test_image)
print(f'Predicted Gesture: {result}')
