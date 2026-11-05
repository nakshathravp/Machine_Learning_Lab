
import numpy as np
import string
import random
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

# Set random seed
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Step 1: Generate CAPTCHA dataset

characters = string.ascii_uppercase + string.digits

# Try different fonts
font_paths = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/calibri.ttf"
]

fonts = []

for path in font_paths:
    try:
        fonts.append(ImageFont.truetype(path, 22))
    except OSError:
        pass

if not fonts:
    fonts = [ImageFont.load_default()]

X, y = [], []

# Generate 200 images for each character
for label, char in enumerate(characters):
    for i in range(200):

        img = Image.new("L", (32, 32), 255)
        draw = ImageDraw.Draw(img)

        font = random.choice(fonts)

        # Random character position
        x = random.randint(3, 8)
        y_pos = random.randint(2, 8)

        draw.text((x, y_pos), char, font=font, fill=0)

        # Rotate image slightly
        img = img.rotate(
            random.randint(-10, 10),
            fillcolor=255
        )

        # Add small amount of noise
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 8, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

        X.append(arr)
        y.append(label)

print("Dataset Generated:", len(X))

# Step 2: Preprocess images

X = np.array(X).reshape(-1, 32, 32, 1) / 255.0
y = np.array(y)

# Step 3: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))

# Step 4: Build CNN model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(32, 32, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(36, activation='softmax')
])

# Step 5: Compile model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Train model

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Step 7: Evaluate model

loss, accuracy = model.evaluate(X_test, y_test)

print("\nMODEL EVALUATION")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# Step 8: Predict character

prediction = model.predict(X_test[:1], verbose=0)

predicted_char = characters[np.argmax(prediction)]
actual_char = characters[y_test[0]]

print("\nCAPTCHA RECOGNITION RESULT")
print("Actual Character:", actual_char)
print("Predicted Character:", predicted_char)

if actual_char == predicted_char:
    print("Character Recognized Successfully")
else:
    print("Incorrect Prediction")

# Step 9: Save model

model.save("captcha_cnn_model.keras")

print("\nCNN Model Saved Successfully!")
