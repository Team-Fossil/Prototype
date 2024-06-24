import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# CSV 파일 로드
data_path = '/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/1.데이터 쌓기/ROCK.csv'
data = pd.read_csv(data_path, header=None)  # 첫 번째 행을 헤더로 사용하지 않음

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

# 시각화를 위해 첫 번째 학습 데이터 샘플을 가져옴
sample_index = 0
sample_data = X_train[sample_index].squeeze()

# sample_data 값 확인
print("Sample Data:", sample_data)

# 첫 번째 샘플 데이터 시각화
plt.figure(figsize=(10, 4))
plt.plot(sample_data)
plt.title('First Sample Data (Preprocessed)')
plt.xlabel('Feature Index')
plt.ylabel('Angle Value')
plt.grid(True)
plt.show()

# 레이블 시각화
print(f"Sample Label: {y_train[sample_index]}")
