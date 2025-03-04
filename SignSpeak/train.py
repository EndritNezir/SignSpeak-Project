from utils.dataset import preprocess_dataset
from models.gesture_model import create_model
import tensorflow as tf

# Dataset Load 
X_train, X_val, y_train, y_val, gesture_dict = preprocess_dataset()

# Model Create 
model = create_model(input_shape=(128, 128, 3), num_classes=len(gesture_dict))

# Model Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Train 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Model Save
tf.keras.models.save_model(model, 'model.keras')