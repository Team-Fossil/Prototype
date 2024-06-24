import numpy as np
import pandas as pd
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
    weights = []
    for layer in model.layers:
        layer_weights = []
        for weight in layer.weights:
            weights_shape = weight.shape
            weights_size = np.prod(weights_shape)
            layer_weights.append(weights_array[start:start + weights_size].reshape(weights_shape))
            start += weights_size
        if layer_weights:
            weights.append(layer_weights)
    
    for layer, weight in zip(model.layers, weights):
        layer.set_weights(weight)

input_shape = 21  # 각도 데이터의 길이 (Wrist부터 Pinky_Tip까지의 21개 랜드마크)
model = create_model(input_shape)
load_weights_from_csv(model, '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/rock_gesture_model_weights.csv')
