import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# CSV 파일 읽기
data_rock = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/ROCK.csv')
data_paper = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/PAPER.csv')
data_scissors = pd.read_csv('/Users/apple/Desktop/Python/2nd_Grade/Commpetition/Prototype/VERSION2/SCISSORS.csv')

# 필요한 열만 선택하여 numpy 배열로 변환
features = ['Wrist',
            'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
            'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
            'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
            'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
            'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']

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

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred, labels=['ROCK', 'PAPER', 'SCISSORS'])

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ROCK', 'PAPER', 'SCISSORS'], yticklabels=['ROCK', 'PAPER', 'SCISSORS'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()