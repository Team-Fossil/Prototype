import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

# CNN 모델 설계
def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape, 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류 ('ROCK' 또는 '기타')
    ])
    return model

# 모델 가중치를 CSV 파일로부터 로드하여 설정
def load_weights_from_csv(model, weights_path):
    weights_df = pd.read_csv(weights_path)
    weights_array = weights_df.values.flatten()
    
    start = 0
    for layer in model.layers:
        layer_weights = []
        for weight in layer.weights:
            weights_shape = weight.shape
            weights_size = np.prod(weights_shape)
            layer_weights.append(weights_array[start:start + weights_size].reshape(weights_shape))
            start += weights_size
        if layer_weights:
            layer.set_weights(layer_weights)

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 모델 생성 및 가중치 로드
input_shape = 21  # 각도 데이터의 길이 (Wrist부터 Pinky_Tip까지의 21개 랜드마크)
model = create_model(input_shape)
load_weights_from_csv(model, '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/rock_gesture_model_weights.csv')

# 웹캠 초기화
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
                    angle_rad = math.atan2(math.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2), x2 - x1)
                    angle_deg = round(math.degrees(angle_rad), 2)
                    flexion_angles.append(int(angle_deg))

                # 각도 배열을 모델 입력 형태로 변환
                input_data = np.expand_dims(np.array(flexion_angles), axis=(0, -1))

                # 예측
                prediction = model.predict(input_data)
                is_rock = prediction[0][0] > 0.5

                # 결과 출력
                text = "ROCK" if is_rock else "NOT ROCK"
                cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()
