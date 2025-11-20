# Real-Time Hand Gesture Controller

Real-time hand gesture recognition system using **MediaPipe**, **TensorFlow/Keras**, **Flask**, and **JavaScript webcam streaming**.

This project captures 21 hand keypoints using MediaPipe (x, y, z → total 63 features) and classifies gestures (rock / paper / scissors) using a lightweight neural network.

---

## Features

### Real-Time Gesture Recognition

* 21 MediaPipe hand landmarks
* Extract (x, y, z) → **63-dimensional feature vector**
* TensorFlow/Keras classification model
* Live webcam streaming using JavaScript
* Fast REST inference server using Flask

### Custom Dataset

* You collect your own gestures using a Python script
* Each gesture is stored as a `.npy` file
* Model trained on personalized hand movements

### Web Interface

* Browser webcam capture
* Sends frames to Flask server
* Displays predicted gesture in real-time

---

## Project Structure

```
RealTime-HandGesture-Controller/
│
├── app.py                  # Flask server (real-time prediction)
├── create_dataset.py       # Collect hand gestures
├── train_model.py          # Train 63-feature Keras model
│
├── model/
│   ├── dataset/            # rock.npy / paper.npy / scissors.npy
│   ├── gesture_model.h5    # trained classifier
│   └── labels.npy          # label array
│
├── static/
│   ├── script.js           # webcam → server streaming
│   └── style.css           # basic UI
│
└── templates/
    └── index.html          # web UI
```

---

## 1. Install Dependencies

### Create environment

```bash
conda create -n handgesture python=3.11
conda activate handgesture
```

### Install packages

```bash
pip install mediapipe opencv-python numpy flask tensorflow
```

---

## 2. Collect Dataset

Run:

```bash
python create_dataset.py
```

You will be prompted:

```
=== Collecting data for: rock ===
Show the gesture and press ENTER to finish.
```

Repeat for:

* rock
* paper
* scissors

Files saved as:

```
model/dataset/rock.npy
model/dataset/paper.npy
model/dataset/scissors.npy
```

Each file contains **63-dimensional vectors** extracted frame-by-frame.

---

## 3. Train Model

Run:

```bash
python train_model.py
```

This generates:

```
model/gesture_model.h5
model/labels.npy
```

---

## 4. Run Real-Time Server

Start Flask:

```bash
python app.py
```

Server runs on:

```
http://localhost:5050
```

Open your browser → webcam feed will start → gesture prediction appears in real-time.

---

## 5. How It Works (Pipeline)

### 1) Frontend

* JavaScript captures webcam frames
* Converts to JPEG blob
* Sends to Flask `/predict` every 200ms

### 2) Backend

* Flask decodes the frame
* MediaPipe extracts 21 hand landmarks
* Creates feature vector (63 dims)
* Keras model predicts gesture
* JSON returned to browser

### 3) UI

* Predicted gesture displayed instantly

---

## Example Output

```
Gesture: rock
Gesture: paper
Gesture: scissors
Gesture: none (no hand detected)
```

---

## Model Input Format

```
[ x1, y1, z1, x2, y2, z2, ... x21, y21, z21 ]  # total 63 dims
```

This input is consistent between:

* dataset collection
* model training
* real-time prediction

---

## Notes / Debug Tips

* If gesture prediction returns `"none"` → check if MediaPipe detected a hand
* Ensure gesture_model.h5 expects 63 features
* If webcam shows black → allow camera permissions

---

## License

MIT License.


