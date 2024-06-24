import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# CSV 파일 로드
rock_data = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/ROCK.csv')

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

# 형태 확인
print(X_train_expanded.shape, X_val_expanded.shape, y_train.shape, y_val.shape)
