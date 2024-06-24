import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# CSV 파일 로드
data_path = '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/1.데이터 쌓기/ROCK.csv'
data = pd.read_csv(data_path, header=None)

# 첫 번째 열(날짜) 제거
data = data.iloc[:, 1:]

# 첫 번째 행(각도 인덱스) 제거
data = data.iloc[1:]

# 데이터 타입을 float으로 변환
data = data.astype(float)

# 결측값을 0으로 대체 (또는 다른 적절한 값으로 대체)
data.fillna(0, inplace=True)

# 데이터와 레이블 분리 (여기서는 레이블이 '0'이라고 가정)
X = data.values
y = np.zeros((X.shape[0],))  # '주먹' 레이블을 0으로 설정

# 데이터셋을 학습 및 테스트셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 차원 확장 (CNN 입력 형태에 맞게)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# CNN 모델 설계
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 이진 분류 ('ROCK' 또는 '기타')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 모델 저장
model_save_path = '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/rock_gesture_model.h5'
model.save(model_save_path)
