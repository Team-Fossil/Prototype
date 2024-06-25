import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# CSV 파일 읽기
data_rock = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION3/ROCK.csv')
data_paper = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION3/PAPER.csv')
data_scissors = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION3/SCISSORS.csv')

# 필요한 열만 선택하여 numpy 배열로 변환
rock_data = data_rock[['Wrist',
                   'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
                   'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
                   'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
                   'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
                   'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']].values

paper_data = data_paper[['Wrist',
                   'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
                   'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
                   'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
                   'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
                   'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']].values

scissors_data = data_scissors[['Wrist',
                   'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
                   'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
                   'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
                   'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
                   'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']].values

# 레이블 추가
rock_labels = np.full(len(rock_data), 'ROCK')
paper_labels = np.full(len(paper_data), 'PAPER')
scissors_labels = np.full(len(scissors_data), 'SCISSORS')

# 데이터 결합
X = np.vstack((rock_data, paper_data, scissors_data))
y = np.concatenate((rock_labels, paper_labels, scissors_labels))

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM 모델 학습
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                flexion_angles = []
                joint_ids = [
                    mp_hands.HandLandmark.WRIST,
                    mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP
                ]
                for i in range(len(joint_ids) - 1):
                    idx1, idx2 = joint_ids[i], joint_ids[i+1]
                    x1, y1, z1 = hand_landmarks.landmark[idx1].x, hand_landmarks.landmark[idx1].y, hand_landmarks.landmark[idx1].z
                    x2, y2, z2 = hand_landmarks.landmark[idx2].x, hand_landmarks.landmark[idx2].y, hand_landmarks.landmark[idx2].z
                    angle_rad = math.atan2(math.sqrt((y2-y1)**2 + (z2-z1)**2), x2-x1)
                    angle_deg = round(math.degrees(angle_rad), 2)
                    flexion_angles.append(int(angle_deg))

                flexion_angles = np.array(flexion_angles).reshape(1, -1)

                # 예측
                prediction = model.predict(flexion_angles)[0]
                decision_function = model.decision_function(flexion_angles)
                max_score = np.max(decision_function)

                # 결과 출력
                threshold = 0.9  # 임계값 설정
                if max_score < threshold:
                    prediction = 'UNKNOWN'

                cv2.putText(image, prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
