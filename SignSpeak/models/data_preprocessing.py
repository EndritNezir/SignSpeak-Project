# moduls/dataset_preprocessing.py

import cv2
import os
import numpy as np

def load_and_preprocess(dataset_dir):
    gestures = os.listdir(dataset_dir)
    images = []
    labels = []

    for gesture in gestures:
        gesture_dir = os.path.join(dataset_dir, gesture)
        if os.path.isdir(gesture_dir):
            for image_name in os.listdir(gesture_dir):
                img_path = os.path.join(gesture_dir, image_name)
                img = cv2.imread(img_path)

                # Preprocessing steps (resize, normalize)
                img_resized = cv2.resize(img, (128, 128))
                img_normalized = img_resized.astype("float32") / 255.0

                images.append(img_normalized)
                labels.append(gesture)  # Gesture as label

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
