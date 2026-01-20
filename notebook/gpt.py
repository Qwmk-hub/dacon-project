from pathlib import Path

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


# =========================
# 1) Load
# =========================
train_path = r"C:\Users\solba\dacon-project\data\raw\train.csv"
test_path  = r"C:\Users\solba\dacon-project\data\raw\test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# =========================
# 2) Drop columns (train/test 동일)
# =========================
columns_to_drop = [
    "generation",
    "contest_award",
    "completed_semester",
    "incumbents_lecture_scale_reason",
    "interested_company",
    "contest_participation",
    "idea_contest",
]

train = train.drop(columns=columns_to_drop, errors="ignore")
test = test.drop(columns=columns_to_drop, errors="ignore")

# =========================
# 3) Missing values -> "Unknown" (문자열로 통일)
# =========================
# CatBoost는 category/str을 잘 처리하니까, 결측은 먼저 Unknown으로 채우고
# 그 다음 category로 바꾸는 게 깔끔함.
for col in train.columns:
    if col in ["ID", "completed"]:
        continue
    train[col] = train[col].astype(str)
    train[col] = train[col].replace("nan", "Unknown").fillna("Unknown")

for col in test.columns:
    if col == "ID":
        continue
    test[col] = test[col].astype(str)
    test[col] = test[col].replace("nan", "Unknown").fillna("Unknown")

# =========================
# 4) (선택) Oversampling
# =========================
# 기존 너 방식 유지: completed=1을 한 번 더 붙임
completed_1 = train[train["completed"] == 1]
train = pd.concat([train, completed_1], ignore_index=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 5) Split X/y
# =========================
X = train.drop(["ID", "completed"], axis=1)
y = train["completed"].astype(int)

X_test = test.drop(["ID"], axis=1)

# 모든 feature를 categorical로 취급 (문자열로 이미 통일했으니 OK)
cat_features = list(range(X.shape[1]))

# =========================
# 6) Model
# =========================
model = CatBoostClassifier(
    iterations=1500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=cat_features,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=200,

    # 불균형 완화 (오버샘플링이 과하면 이 옵션은 빼도 됨)
    auto_class_weights="Balanced",

    # GPU
    task_type="GPU",
    devices="0"
)

print("모델 학습 시작... (GPU 모드)")
model.fit(X, y)
print("모델 학습 완료!")

# =========================
# 7) Train Eval (분류 지표)
# =========================
train_proba_all = model.predict_proba(X)
# completed=1 확률 컬럼 인덱스 (classes_ 순서가 뒤집혀도 안전)
pos_idx = list(model.classes_).index(1)
train_proba = train_proba_all[:, pos_idx]

train_pred = (train_proba >= 0.5).astype(int)

print("\n[Train Metrics]")
print("classes_:", model.classes_)
print(f"Train AUC     : {roc_auc_score(y, train_proba):.6f}")
print(f"Train LogLoss : {log_loss(y, train_proba):.6f}")
print(f"Train Acc     : {accuracy_score(y, train_pred):.6f}")

# =========================
# 8) Test Predict + Save submission
# =========================
test_proba_all = model.predict_proba(X_test)
test_proba = test_proba_all[:, pos_idx]  # completed=1 확률

# 기본 임계값 0.5
test_pred = (test_proba >= 0.5).astype(int)

submission = pd.DataFrame({
    "ID": test["ID"],
    "completed": test_pred
})

out_dir = Path(r"C:\Users\solba\dacon-project\output")
out_dir.mkdir(parents=True, exist_ok=True)

submission_path = out_dir / "submission_catboost.csv"
submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 제출 파일 저장 완료: {submission_path}")
print(submission.head())

# (선택) 확률도 같이 보고 싶으면 아래 저장
proba_path = out_dir / "test_with_proba.csv"
pd.DataFrame({"ID": test["ID"], "proba_completed_1": test_proba}).to_csv(
    proba_path, index=False, encoding="utf-8-sig"
)
print(f"✅ 확률 파일 저장 완료: {proba_path}")
