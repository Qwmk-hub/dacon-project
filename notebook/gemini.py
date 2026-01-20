from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. 데이터 로드
train = pd.read_csv(r"C:\Users\solba\dacon-project\data\raw\train.csv")
test = pd.read_csv(r"C:\Users\solba\dacon-project\data\raw\test.csv")

# 2. 필요없는 칼럼 Drop
columns_to_drop = ['generation', 'contest_award', 'completed_semester', 
                   'incumbents_lecture_scale_reason', 'interested_company', 
                   'contest_participation', 'idea_contest']
train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)

# 3. ID 분리
train_x = train.drop(['ID', 'completed'], axis=1)
train_y = train['completed']
test_x = test.drop(['ID'], axis=1)
test_ids = test['ID']

# 4. 범주형/수치형 변수 자동 구분 및 처리
# (수동으로 지정하는 것이 가장 좋지만, 자동화를 위해 타입으로 구분)
cat_features = []
for col in train_x.columns:
    # 데이터 타입이 object(문자열)이거나 유니크 값이 적은 경우 범주형으로 간주
    if train_x[col].dtype == 'object':
        cat_features.append(col)
        # 결측치 처리 (범주형만)
        train_x[col] = train_x[col].fillna('Unknown').astype(str)
        test_x[col] = test_x[col].fillna('Unknown').astype(str)
    else:
        # 수치형 결측치는 그대로 둡니다 (CatBoost가 처리함) 또는 0/-1 등으로 대체
        pass

print(f"범주형 변수: {cat_features}")

# 5. 검증 데이터셋 분리 (중요!)
X_train, X_val, y_train, y_val = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
)

# 6. 모델 정의 (오버샘플링 대신 class_weights 사용)
model = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    loss_function='Logloss',    
    eval_metric='AUC',        
    cat_features=cat_features, # 실제 범주형 컬럼 이름 리스트 전달
    auto_class_weights='Balanced', # 클래스 불균형 자동 해결
    random_seed=42,
    verbose=200,
    task_type="GPU", # GPU 없으면 "CPU"로 변경
    devices='0',
    early_stopping_rounds=100 # 과적합 방지
)

print("모델 학습 시작...")

# 7. 학습 (검증 데이터 포함)
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

print("\n모델 학습 완료!")

# 8. 검증 데이터 평가
val_pred = model.predict(X_val)
val_proba = model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, val_pred)
f1 = f1_score(y_val, val_pred)
auc = roc_auc_score(y_val, val_proba)

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print(f"Validation AUC: {auc:.4f}")

# 9. Test 예측 및 저장
test_pred = model.predict(test_x)

submission = pd.DataFrame({
    'ID': test_ids,
    'completed': test_pred
})

output_dir = Path(r"C:\Users\solba\dacon-project\output")
output_dir.mkdir(parents=True, exist_ok=True)
submission_path = output_dir / "submission_catboost_fixed.csv"
submission.to_csv(submission_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 저장 완료: {submission_path}")