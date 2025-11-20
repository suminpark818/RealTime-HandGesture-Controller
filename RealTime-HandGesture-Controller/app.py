from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model (63 input dims)
model = load_model("model/gesture_model.h5")
labels = np.load("model/labels.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # No hand → none
    if not result.multi_hand_landmarks:
        return jsonify({"gesture": "none"})

    # Use the first detected hand
    hand_landmarks = result.multi_hand_landmarks[0]

    # Extract 63 features (x, y, z)
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])

    # Ensure correct length (63)
    if len(keypoints) != 63:
        return jsonify({"gesture": "none"})

    # Predict
    pred = model.predict(np.array([keypoints]))[0]
    idx = np.argmax(pred)
    gesture = labels[idx]

    return jsonify({"gesture": gesture})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
