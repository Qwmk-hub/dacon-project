from __future__ import annotations

import os
import importlib
import numpy as np
import pandas as pd

import Alpha
importlib.reload(Alpha)
from Alpha import merge_df, completed

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# =========================
# Config
# =========================
TRAIN_PATH = r"C:\Users\solba\dacon-project\data\raw\train.csv"
TEST_PATH  = r"C:\Users\solba\dacon-project\data\raw\test.csv"
OUT_PATH   = r"C:\Users\solba\dacon-project\output\submit_catboost.csv"

ID_COL = "ID"
TARGET = "completed"
SEED = 42

N_SPLITS = 5
EARLY_STOPPING = 200

THR = 0.447

# =========================
# Load
# =========================
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

# =========================
# Feature engineering
# =========================
train_df = merge_df(train_raw)
test_df  = merge_df(test_raw)

# (선택) completed 함수 사용: 이미 너 파이프라인에서 쓰고 있으니 유지
train_df = completed(train_raw, train_df)

# =========================
# Copy + sort (row alignment)
# =========================
train = train_df.copy()
test = test_df.copy()

train = train.sort_values(ID_COL).reset_index(drop=True)
test  = test.sort_values(ID_COL).reset_index(drop=True)

# =========================
# Feature definition
# =========================
cat_cols = [
    "school1", "job", "nationality", "High Tech", "Data Friendly", "Others",
    "hope_for_group", "desired_career_path",
    "incumbents_level", "incumbents_lecture", "incumbents_company_level",
    "incumbents_lecture_type", "incumbents_lecture_scale",
]
num_cols = ["count", "time_input", "want_count"]

# Safety checks
for c in [ID_COL] + cat_cols + num_cols:
    if c not in train.columns:
        raise ValueError(f"[train] Column '{c}' not found.")
for c in [ID_COL] + cat_cols + num_cols:
    if c not in test.columns:
        raise ValueError(f"[test] Column '{c}' not found.")
if TARGET not in train.columns:
    raise ValueError(f"[train] Target '{TARGET}' not found. completed()가 타겟을 못 붙였을 수 있음.")

# Build X/y
X = train[cat_cols + num_cols].copy()
y = train[TARGET].astype(int).reset_index(drop=True)
X_test = test[cat_cols + num_cols].copy()

# =========================
# Type handling
# =========================
for c in cat_cols:
    X[c] = X[c].astype("string").fillna("__MISSING__")
    X_test[c] = X_test[c].astype("string").fillna("__MISSING__")

for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# 정합성 체크
assert len(X) == len(y)
assert (X.index.values == y.index.values).all()

# =========================
# Model params (Balanced ON)
# =========================
params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1000,          # 최대 반복 횟수는 유지하되
    learning_rate=0.03,       # 학습률을 조금 올려서 수렴 속도를 확보 (0.01~0.05 사이 추천)
    depth=4,                  # 중요: 5 -> 3 또는 4로 낮춤 (모델 복잡도 감소)
    l2_leaf_reg=5.0,          # 규제 강도 (3~10 사이 추천, 데이터가 적으면 높게 유지)
    min_data_in_leaf=50,      # 중요: 20 -> 50 (한 잎에 더 많은 데이터가 모이게 하여 노이즈 방지)
    
    # --- 추가/변경된 중요한 파라미터 ---
    rsm=0.8,                  # (colsample_bylevel) 피처의 80%만 무작위로 사용해 과적합 방지
    subsample=0.8,            # 데이터의 80%만 샘플링하여 학습 (Bagging 효과)
    
    od_type="Iter",           # Overfitting Detector 켜기
    od_wait=50,               # 50번 동안 성능 향상이 없으면 조기 종료 (필수)
    
    random_seed=SEED,
    verbose=100,              # 로그 출력 줄임
    allow_writing_files=False,
    auto_class_weights="Balanced",
)

# =========================
# CV training + OOF
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_proba = np.zeros(len(X), dtype=float)
test_proba_folds = []
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    va_pool = Pool(X_va, y_va, cat_features=cat_idx)
    te_pool = Pool(X_test, cat_features=cat_idx)

    model = CatBoostClassifier(**params)
    model.fit(
        tr_pool,
        eval_set=va_pool,
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING,
    )

    va_pred = model.predict_proba(va_pool)[:, 1]
    oof_proba[va_idx] = va_pred

    va_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(va_auc)

    test_proba_folds.append(model.predict_proba(te_pool)[:, 1])

    print(f"[Fold {fold}] AUC={va_auc:.5f} | best_iter={model.get_best_iteration()}")

print("\nCV AUC mean±std:", float(np.mean(fold_scores)), "±", float(np.std(fold_scores)))
print("OOF AUC:", float(roc_auc_score(y, oof_proba)))

# =========================
# Diagnostics: proba distribution + OOF metrics @ THR
# =========================
print("\nOOF proba summary:",
      float(np.min(oof_proba)),
      float(np.mean(oof_proba)),
      float(np.max(oof_proba)))
print(f"OOF proba >= {THR}: {(oof_proba >= THR).sum()} / {len(oof_proba)}")

pred_oof = (oof_proba >= THR).astype(int)
acc = float(accuracy_score(y, pred_oof))
prec = float(precision_score(y, pred_oof, zero_division=0))
rec = float(recall_score(y, pred_oof, zero_division=0))
f1 = float(f1_score(y, pred_oof, zero_division=0))
print(f"OOF metrics @thr={THR:.2f} | ACC={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}")

# =========================
# Predict test + apply threshold 0.45 + save
# =========================
test_proba = np.mean(np.vstack(test_proba_folds), axis=0)

print("\nTEST proba summary:",
      float(np.min(test_proba)),
      float(np.mean(test_proba)),
      float(np.max(test_proba)))
print(f"TEST proba >= {THR}: {(test_proba >= THR).sum()} / {len(test_proba)}")

test_pred = (test_proba >= THR).astype(int)

submit = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_pred
})

print("\nsubmit value counts:", submit[TARGET].value_counts(dropna=False).to_dict())

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
submit.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH} (threshold={THR})")
