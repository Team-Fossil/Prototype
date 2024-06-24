import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

# CSV 파일 경로
csv_path = '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/ROCK.csv'

# CSV 파일 로드
rock_data = pd.read_csv(csv_path)

# NaN 값 처리 (여기서는 NaN 값을 0으로 대체)
rock_data.fillna(0, inplace=True)

# 입력 데이터 (X)와 출력 데이터 (y) 분리
X = rock_data.drop(columns=['Timestamp'])
y = np.ones(len(X))  # 'ROCK' 제스처를 모두 1로 라벨링

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터셋 분리
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 데이터를 3차원으로 변환
X_train_expanded = np.expand_dims(X_train, axis=-1)
X_val_expanded = np.expand_dims(X_val, axis=-1)

# CNN 모델 정의
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_expanded.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train_expanded, y_train, epochs=20, validation_data=(X_val_expanded, y_val))

# 학습 결과 출력
print(history.history)
