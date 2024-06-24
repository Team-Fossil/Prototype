import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt

# 모델 로드
model_path = '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/my_model.keras'
model = load_model(model_path)

# 데이터 로드 및 전처리
csv_path = '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/ROCK.csv'
rock_data = pd.read_csv(csv_path)
rock_data.fillna(0, inplace=True)
X = rock_data.drop(columns=['Timestamp'])
y = np.ones(len(X))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_expanded = np.expand_dims(X_train, axis=-1)
X_val_expanded = np.expand_dims(X_val, axis=-1)

# 모델에 가상의 입력을 한 번 실행하여 입력 텐서를 정의합니다
_ = model.predict(X_train_expanded[:1])

# 중간 레이어의 출력을 얻기 위한 모델을 정의합니다
layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# 활성화 맵 시각화
activations = activation_model.predict(X_train_expanded[:1])
layer_names = [layer.name for layer in model.layers[:6]]

for layer_name, layer_activation in zip(layer_names, activations):
    if len(layer_activation.shape) == 3:  # Conv1D 레이어의 경우
        fig, axes = plt.subplots(1, min(8, layer_activation.shape[-1]), figsize=(20, 4))
        for i in range(min(8, layer_activation.shape[-1])):
            ax = axes[i]
            ax.plot(layer_activation[0, :, i])
            ax.set_title(f'{layer_name} - Filter {i}')
        plt.show()
