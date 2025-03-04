# hand_dataset_collector.py

import cv2
import os
import numpy as np

dataset_dir = '../asl_dataset'  # Path to your downloaded dataset

# Function to preprocess dataset
def preprocess_dataset():
    gestures = os.listdir(dataset_dir)
    images = []
    labels = []

    for gesture in gestures:
        gesture_dir = os.path.join(dataset_dir, gesture)
        if os.path.isdir(gesture_dir):
            for image_name in os.listdir(gesture_dir):
                img_path = os.path.join(gesture_dir, image_name)
                img = cv2.imread(img_path)
                
                # Resize image to uniform size (128x128)
                img_resized = cv2.resize(img, (128, 128))
                
                # Normalize image
                img_normalized = img_resized.astype("float32") / 255.0
                
                # Append image and its label
                images.append(img_normalized)
                labels.append(gesture)  # Gesture as label

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Print some information about the dataset
    print(f"Total number of gestures: {len(gestures)}")
    print(f"Total number of images: {len(images)}")
    print(f"Image shape: {images.shape[1:]}")  # (128, 128, 3)
    print(f"Unique gestures in dataset: {np.unique(labels)}")

    return images, labels

# Call the function to check dataset
images, labels = preprocess_dataset()

