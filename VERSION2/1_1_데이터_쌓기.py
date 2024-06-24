import cv2
import mediapipe as mp
import math
import time
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# CSV 파일을 열고 writer 객체를 생성합니다.
# 파일이 존재하지 않으면 새로 생성하고, 존재하면 추가로 기록
with open('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/ROCK.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # 파일이 비어있으면 헤더를 작성합니다.
    file.seek(0)
    if file.tell() == 0:
        writer.writerow([
            'Timestamp',
            'Wrist',
            'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
            'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
            'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
            'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
            'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip'
        ])

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        prev_time = time.time()
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
                    # 각 손가락의 굽힘 각도 계산
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
                        flexion_angles.append(int(angle_deg))  # 정수로 변환하여 추가

                    curr_time = time.time()
                    if curr_time - prev_time >= 0.25:  # 0.5초에 한 번 출력 및 CSV에 기록
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(curr_time))
                        writer.writerow([timestamp] + flexion_angles)
                        print(f"{flexion_angles},")  # 배열 뒤에 컴마 추가
                        prev_time = curr_time

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-5)  # 작은 값을 더하여 ZeroDivisionError 방지
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
