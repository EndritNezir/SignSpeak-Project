from utils.dataset import preprocess_dataset
from models.gesture_model import create_model
import tensorflow as tf

# Dataset Load کرو
X_train, X_val, y_train, y_val, gesture_dict = preprocess_dataset()

# Model Create کرو
model = create_model(input_shape=(128, 128, 3), num_classes=len(gesture_dict))

# Model Compile کرو
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Train کرو
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Model Save کرو
tf.keras.saving.save_model(model, 'model.keras')
