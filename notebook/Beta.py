from __future__ import annotations
import pandas as pd
import importlib
import Alpha
importlib.reload(Alpha)
from Alpha import merge_df, completed

import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

train = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\train.csv")
test = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\test.csv")

train_df = merge_df(train)
test_df = merge_df(test)

train_df = completed(train, train_df)

"""
CatBoost binary classification (completed: 0/1)
- Mixed types: categorical(str/ID-like) + numeric(int)
- Stratified K-Fold CV (AUC)
- Out-of-fold probability + threshold tuning for best F1
- Final training on full data + prediction on test

Usage:
  1) Put train.csv / test.csv in DATA_DIR
  2) Adjust FILE names if needed
  3) Run

Notes:
- CatBoost handles categorical features if you pass cat_features indices (or names via Pool).
- We will treat the following as categorical:
  school1, job, nationality, High Tech, Data Friendly, Others,
  hope_for_group, desired_career_path,
  incumbents_level, incumbents_lecture, incumbents_company_level,
  incumbents_lecture_type, incumbents_lecture_scale
- Numeric:
  count, time_input, want_count
"""



# --- after you build train_df/test_df and add completed ---
train = train_df.copy()
test = test_df.copy()

# ✅ 핵심: ID로 정렬해서 row alignment 고정
train = train.sort_values("ID").reset_index(drop=True)
test  = test.sort_values("ID").reset_index(drop=True)

cat_cols = [
    "school1","job","nationality","High Tech","Data Friendly","Others",
    "hope_for_group","desired_career_path",
    "incumbents_level","incumbents_lecture","incumbents_company_level",
    "incumbents_lecture_type","incumbents_lecture_scale",
]
num_cols = ["count","time_input","want_count"]

X = train[cat_cols + num_cols].copy()
y = train["completed"].astype(int).reset_index(drop=True)
X_test = test[cat_cols + num_cols].copy()

# ✅ 타입 처리
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

# ✅ OOF 정합성 체크
assert len(X) == len(y)
assert (X.index.values == y.index.values).all()

params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=20000,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=10.0,
    min_data_in_leaf=20,
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    auto_class_weights="Balanced",  # 불균형이면 꽤 도움
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
        early_stopping_rounds=600,  # 더 느슨하게
    )

    va_pred = model.predict_proba(va_pool)[:, 1]
    oof_proba[va_idx] = va_pred

    va_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(va_auc)
    test_proba_folds.append(model.predict_proba(te_pool)[:, 1])

    print(f"[Fold {fold}] AUC={va_auc:.5f} | best_iter={model.get_best_iteration()}")

print("\nCV AUC mean±std:", float(np.mean(fold_scores)), "±", float(np.std(fold_scores)))
print("OOF AUC:", roc_auc_score(y, oof_proba))
