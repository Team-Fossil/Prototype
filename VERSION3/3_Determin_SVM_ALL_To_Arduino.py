# 데이터 분석 및 처리를 위해 pandas와 numpy를 import
import pandas as pd
import numpy as np

# 머신러닝 모델 학습과 평가를 위해 필요한 라이브러리 import
from sklearn.model_selection import train_test_split  # 데이터셋 분할
from sklearn.svm import SVC  # SVM 모델
from sklearn.preprocessing import StandardScaler  # 데이터 표준화
from sklearn.pipeline import make_pipeline  # 파이프라인 생성
from sklearn.metrics import accuracy_score  # 정확도 계산

# OpenCV와 MediaPipe를 사용하여 실시간 비디오 처리 및 손동작 인식
import cv2
import mediapipe as mp

# 수학 연산을 위한 math 라이브러리 import
import math

# 시리얼 통신을 위한 serial 라이브러리 import
import serial

# 시간 제어를 위한 time 라이브러리 import
import time

# CSV 파일 읽기
data_rock = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION3/ROCK.csv')
data_paper = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION3/PAPER.csv')
data_scissors = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION3/SCISSORS.csv')

# 필요한 열만 선택하여 numpy 배열로 변환
# Timestamp와 NAN 값들로 있는 Pinky_TIP은 제외
features = ['Wrist', 'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
            'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
            'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
            'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
            'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']

# 각 데이터셋에서 필요한 열만 선택하여 numpy 배열로 변환
rock_data = data_rock[features].values
paper_data = data_paper[features].values
scissors_data = data_scissors[features].values

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

# 시리얼 포트 설정
ser = serial.Serial('/dev/tty.usbserial-0001', 9600)  # Mac에서 시리얼 포트 경로 설정
time.sleep(2)  # 시리얼 포트가 초기화될 시간을 줍니다.

# MediaPipe 및 OpenCV 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 카메라 설정
cap = cv2.VideoCapture(0)

last_sent_time = time.time()  # 마지막으로 전송한 시간

# MediaPipe 손 모델 초기화
with mp_hands.Hands(
    model_complexity=0,  # 모델 복잡도 설정 (0: 가벼운 모델)
    min_detection_confidence=0.5,  # 최소 탐지 신뢰도
    min_tracking_confidence=0.5) as hands:  # 최소 추적 신뢰도

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        # 이미지 전처리
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                flexion_angles = []  # 각 관절의 굽힘 각도를 저장할 리스트
                joint_ids = [
                    mp_hands.HandLandmark.WRIST,
                    mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP
                ]
                
                # 각 관절 사이의 각도를 계산
                for i in range(len(joint_ids) - 1):
                    idx1, idx2 = joint_ids[i], joint_ids[i+1]
                    x1, y1, z1 = hand_landmarks.landmark[idx1].x, hand_landmarks.landmark[idx1].y, hand_landmarks.landmark[idx1].z
                    x2, y2, z2 = hand_landmarks.landmark[idx2].x, hand_landmarks.landmark[idx2].y, hand_landmarks.landmark[idx2].z
                    angle_rad = math.atan2(math.sqrt((y2-y1)**2 + (z2-z1)**2), x2-x1)
                    angle_deg = round(math.degrees(angle_rad), 2)
                    flexion_angles.append(int(angle_deg))

                # 예측을 위해 배열 형태로 변환
                flexion_angles = np.array(flexion_angles).reshape(1, -1)

                # 예측
                prediction = model.predict(flexion_angles)[0]
                decision_function = model.decision_function(flexion_angles)
                max_score = np.max(decision_function)

                # 결과 출력
                threshold = 2.2  # 임계값 설정
                if max_score < threshold:
                    prediction = 'UNKNOWN'

                # 이미지에 예측된 결과를 표시
                cv2.putText(image, prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # 1초에 한 번씩 예측된 결과를 아두이노로 전송
                current_time = time.time()
                if current_time - last_sent_time >= 1.0:
                    ser.write((prediction + '\n').encode())
                    last_sent_time = current_time  # 마지막으로 전송한 시간 갱신
                    if ser.in_waiting > 0:
                        received_data = ser.readline().decode('utf-8').strip()
                        print(f"Received from Arduino: {received_data}")

        # 이미지 출력
        cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # 'ESC' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
ser.close()