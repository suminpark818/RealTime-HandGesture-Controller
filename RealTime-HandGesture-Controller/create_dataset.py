import cv2
import mediapipe as mp
import numpy as np
import os

# 클래스 순서
class_names = ["rock", "paper", "scissors"]

# 저장 경로
dataset_path = "model/dataset"
os.makedirs(dataset_path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def collect_for_class(label):
    print(f"\n=== Collecting data for: {label} ===")
    print("Show the gesture and press ENTER to finish.\n")

    cap = cv2.VideoCapture(0)
    data = []

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # 21 keypoints (x,y,z)
                lm_list = []
                for lm in handLms.landmark:
                    lm_list.extend([lm.x, lm.y, lm.z])
                data.append(lm_list)

        cv2.putText(img,
                    f"Collecting: {label} | Frames: {len(data)}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Dataset Collection", img)

        # ENTER → break
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

    data = np.array(data)
    np.save(os.path.join(dataset_path, f"{label}.npy"), data)
    print(f"Saved {data.shape[0]} frames for class '{label}'")


for label in class_names:
    collect_for_class(label)

print("\n=== Dataset Collection Completed ===")
