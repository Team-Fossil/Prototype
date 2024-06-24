import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# 모델의 첫 번째 Conv1D 레이어의 가중치 시각화
conv1_weights = model.layers[0].get_weights()[0]
print(conv1_weights.shape)  # (3, 1, 64) 등의 형태

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < conv1_weights.shape[-1]:
        filter_weights = conv1_weights[:, :, i]
        ax.plot(filter_weights)
        ax.set_xticks([])
        ax.set_yticks([])
plt.suptitle('Conv1D Layer Filters')
plt.show()
