import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

dataset_dir = './Guesture_dataset'  

# Function to preprocess dataset
def preprocess_dataset():
    gestures = sorted(os.listdir(dataset_dir))
    images, labels = [], []
    
    for idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(dataset_dir, gesture)
        if os.path.isdir(gesture_dir):
            for image_name in os.listdir(gesture_dir):
                img_path = os.path.join(gesture_dir, image_name)
                img = cv2.imread(img_path)
                
                if img is not None:  # Avoid errors due to unreadable images
                    img_resized = cv2.resize(img, (128, 128))
                    img_normalized = img_resized.astype("float32") / 255.0
                    
                    images.append(img_normalized)
                    labels.append(idx)  # Numeric label for each gesture

    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels, num_classes=len(gestures))

    # Split into training (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, gestures

# Run and check dataset loading
if __name__ == "__main__":
    X_train, X_val, y_train, y_val, gestures = preprocess_dataset()
    print(f"Total gestures: {len(gestures)}")
    print(f"Total training images: {X_train.shape[0]}")
    print(f"Total validation images: {X_val.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")
