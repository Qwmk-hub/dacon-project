from pathlib import Path

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


train = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\train.csv")
test = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\test.csv")

# 필요없는 칼럼 Drop
columns_to_drop = ['generation', 'contest_award', 'completed_semester', 
                   'incumbents_lecture_scale_reason', 'interested_company', 'contest_participation', 'idea_contest']

train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)


# 카테고리 변환
cat_cols = []
for column in train.columns:
    if column == 'ID' or column == 'completed':
        continue
    
    train[column] = train[column].astype(str).astype('category')
    if column in test.columns:
        test[column] = test[column].astype(str).astype('category')
    
    cat_cols.append(column)

# 결측치 처리: 'Unknown'으로 대체
for column in train.columns:
    if train[column].isnull().sum() > 0:
        train[column].fillna('Unknown', inplace=True)

for column in test.columns:
    if test[column].isnull().sum() > 0:
        test[column].fillna('Unknown', inplace=True)

# 클래스 불균형 해결을 위한 오버샘플링
completed_1 = train[train['completed'] == 1]
train_oversampled = pd.concat([train, completed_1], ignore_index=True)
train_oversampled = train_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
train = train_oversampled

# ID 열 제외하고 X, y 분리
X = train.drop(['ID', 'completed'], axis=1)
y = train['completed']

baseline_values = np.zeros(len(train))

for col in X.columns:
    category_means = train.groupby(col)['completed'].mean()
    
    for idx in range(len(train)):
        category_value = train[col].iloc[idx]
        baseline_values[idx] += category_means[category_value]

baseline_values = baseline_values / len(X.columns)

cat_features = list(range(len(X.columns)))

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.03,

    loss_function='Logloss',    
    eval_metric='AUC',        

    cat_features=cat_features,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,

    # GPU
    task_type="GPU",
    devices='0'
)


print("모델 학습 시작... (GPU 모드)")

model.fit(
    X, y,
    baseline=baseline_values, 
    verbose=200
)

print("\n모델 학습 완료!")

# 예측 및 평가
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

y_pred_binary = (y_pred >= 0.5).astype(int)
accuracy = accuracy_score(y, y_pred_binary)

print(f"Train RMSE: {rmse:.4f}")
print(f"Train R2 Score: {r2:.4f}")
print(f"Train Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


# output_dir = Path(r"C:\Users\solba\dacon-project\output")
# output_dir.mkdir(parents=True, exist_ok=True)
# model_path = output_dir / "catboost_model.cbm"
# model.save_model(model_path)
# print(f"Model saved to {model_path}")

