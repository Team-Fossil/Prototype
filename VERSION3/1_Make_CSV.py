import cv2  # 영상 처리를 위한 OpenCV 라이브러리
import mediapipe as mp  # 손 감지를 위한 Mediapipe 라이브러리
import math  # 수학 함수를 사용하기 위한 math 라이브러리
import time  # 시간 관련 함수를 사용하기 위한 time 라이브러리
import csv  # CSV 파일을 처리하기 위한 csv 라이브러리

# Mediapipe의 모듈 및 클래스 설정
# 손의 랜드마크등 그래픽적인 요소를 그리기 위한 유틸리티 함수
mp_drawing = mp.solutions.drawing_utils
# 기본적인 랜드마크 스타일, 연결 선 스타일 등이 포함
mp_drawing_styles = mp.solutions.drawing_styles
# 손의 감지, 추적, 랜드마크 위치 예측 등을 수행 
mp_hands = mp.solutions.hands

# CSV 파일을 열고 writer 객체를 생성합니다.
with open('TEST.csv', mode='w', newline='') as file:
    writer = csv.writer(file)  # CSV 파일에 쓰기 위한 writer 객체 생성

    # CSV 파일에 헤더를 작성합니다.
    writer.writerow([
        'Timestamp', 'Wrist',
        'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
        'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
        'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
        'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip'
    ])

    # 카메라 오픈
    cap = cv2.VideoCapture(0)

    # Mediapipe Hands 모델 설정
    with mp_hands.Hands(
        model_complexity=0,  # 모델 복잡도 설정 (0: 가벼운 모델)
        min_detection_confidence=0.5,  # 최소 탐지 신뢰도
        min_tracking_confidence=0.5) as hands:  # 최소 추적 신뢰도

        prev_time = time.time()  # 이전 시간 초기화
        while cap.isOpened():  # 카메라가 열려 있는 동안 반복
            success, image = cap.read()  # 프레임 읽기
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # 이미지 뒤집고 RGB로 변환
            results = hands.process(image)  # Mediapipe Hands로 손 감지

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    flexion_angles = []  # 손가락 굽힘 각도를 저장할 리스트 초기화
                    # 각 손가락의 관절 ID 정의
                    joint_ids = [
                        mp_hands.HandLandmark.WRIST,
                        mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP,
                        mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP
                    ]
                    # 각 손가락의 굽힘 각도 계산
		                # hand_landmarks.landmark[idx]를 사용하여 손의 각 관절의 위치를 파악
			              # 이 위치는 손의 이미지 상의 정규화된 좌표로 나타난다. (일반적으로 0에서 1 사이의 값)
										# 각 관절의 (x, y, z) 좌표를 획득한 후, 두 관절 사이의 각도를 계산
										# math.atan2 함수를 사용하여 두 관절 사이의 세로 및 깊이 차이에 대한 아크탄젠트를 계산
										# y2 - y1: 두 관절 사이의 세로(수직) 차이
										# z2 - z1: 두 관절 사이의 깊이(수평) 차이
										# x2 - x1: 두 관절 사이의 가로(수평) 차이
										# 이렇게 계산된 아크탄젠트 값은 라디안 단위
										# 따라서 math.degrees 함수를 사용하여 라디안 값을 도 단위로 변환
                    for i in range(len(joint_ids) - 1):
                        idx1, idx2 = joint_ids[i], joint_ids[i+1]
                        x1, y1, z1 = hand_landmarks.landmark[idx1].x, hand_landmarks.landmark[idx1].y, hand_landmarks.landmark[idx1].z
                        x2, y2, z2 = hand_landmarks.landmark[idx2].x, hand_landmarks.landmark[idx2].y, hand_landmarks.landmark[idx2].z
                        angle_rad = math.atan2(math.sqrt((y2-y1)**2 + (z2-z1)**2), x2-x1)
                        angle_deg = round(math.degrees(angle_rad), 2)
                        flexion_angles.append(int(angle_deg))  # 정수로 변환하여 리스트에 추가

                    curr_time = time.time()  # 현재 시간 기록
                    if curr_time - prev_time >= 0.5:  # 0.5초마다 실행
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(curr_time))
                        writer.writerow([timestamp] + flexion_angles)  # CSV 파일에 타임스탬프와 굽힘 각도 기록
                        print(f"{flexion_angles},")  # 굽힘 각도를 출력 (배열 뒤에 컴마 추가)
                        prev_time = curr_time  # 이전 시간을 현재 시간으로 업데이트

                    mp_drawing.draw_landmarks(  # 손가락 관절에 랜드마크 그리기
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            curr_time = time.time()  # 현재 시간 기록
            fps = 1 / (curr_time - prev_time + 1e-5)  # FPS 계산 (ZeroDivisionError 방지를 위해 작은 값 더함)
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 이미지에 FPS 출력

            cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # 화면에 이미지 출력

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
                break

    cap.release()  # 카메라 해제
    cv2.destroyAllWindows()  # 창 모두 닫기
