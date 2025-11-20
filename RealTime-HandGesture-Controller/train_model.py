import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

DATASET_PATH = "model/dataset"

# Load .npy files
rock = np.load(os.path.join(DATASET_PATH, "rock.npy"))
paper = np.load(os.path.join(DATASET_PATH, "paper.npy"))
scissors = np.load(os.path.join(DATASET_PATH, "scissors.npy"))

# Create labels
y_rock = np.zeros(len(rock))
y_paper = np.ones(len(paper))
y_scissors = np.full(len(scissors), 2)

# Combine
X = np.concatenate([rock, paper, scissors], axis=0)
y = np.concatenate([y_rock, y_paper, y_scissors])

# One-hot labels
y = to_categorical(y, num_classes=3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),  # 21 keypoints × 3 coords
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Save model
model.save("model/gesture_model.h5")
np.save("model/labels.npy", ["rock", "paper", "scissors"])

print("Model training complete. Saved to model/gesture_model.h5")
