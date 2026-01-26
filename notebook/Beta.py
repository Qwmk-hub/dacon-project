from __future__ import annotations
import pandas as pd
import importlib
import Alpha
importlib.reload(Alpha)
from Alpha import merge_df, completed

import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# --------------------
# Load
# --------------------
train_raw = pd.read_csv("/Users/parksung-cheol/Desktop/dacon-project/data/raw/train.csv")
test_raw  = pd.read_csv("/Users/parksung-cheol/Desktop/dacon-project/data/raw/test.csv")

train_df = merge_df(train_raw)
test_df  = merge_df(test_raw)
train_df = completed(train_raw, train_df)

train_df = train_df.sort_values("ID").reset_index(drop=True)
test_df  = test_df.sort_values("ID").reset_index(drop=True)

# --------------------
# Columns
# --------------------
cat_cols = [
    "school1","job","nationality","High Tech","Data Friendly","Others",
    "hope_for_group","desired_career_path",
    "incumbents_level","incumbents_lecture","incumbents_company_level",
    "incumbents_lecture_type","incumbents_lecture_scale",
]
num_cols = ["count","time_input","want_count"]
x_cols   = cat_cols + num_cols
target_col = "completed"

# --------------------
# dtype fix (중요)
# - CatBoost에 cat_features로 넘길 컬럼은 문자열/카테고리로 확실히 만들어주기
# --------------------
for c in cat_cols:
    train_df[c] = train_df[c].astype(str)
    test_df[c]  = test_df[c].astype(str)

# --------------------
# CV (중요: Stratified)
# --------------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

models = []
oof_pred = np.zeros(len(train_df), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(rskf.split(train_df[x_cols], train_df[target_col]), 1):
    X_tr = train_df.loc[tr_idx, x_cols].copy()
    y_tr = train_df.loc[tr_idx, target_col].astype(int).values
    X_va = train_df.loc[va_idx, x_cols].copy()
    y_va = train_df.loc[va_idx, target_col].astype(int).values

    X_te = test_df[x_cols].copy()

    # --------------------
    # Scaling (중요: 누수 방지)
    # - fold의 train으로만 fit
    # - valid/test는 transform만
    # --------------------
    scaler = MinMaxScaler()
    X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
    X_va[num_cols] = scaler.transform(X_va[num_cols])
    X_te[num_cols] = scaler.transform(X_te[num_cols])

    params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1000,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=5.0,
        min_data_in_leaf=20,      # (너 주석이랑 맞춰서 20 권장, 더 빡세게면 30~50)
        rsm=0.8,
        subsample=0.8,
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)  # cat_features는 "컬럼명 리스트"로
    va_pool = Pool(X_va, y_va, cat_features=cat_cols)
    te_pool = Pool(X_te, cat_features=cat_cols)

    model = CatBoostClassifier(**params)
    model.fit(
        tr_pool,
        eval_set=va_pool,
        use_best_model=True,
        early_stopping_rounds=100,  # od_*랑 중복 피하려고 params에서는 od_* 제거
    )

    # OOF
    oof_pred[va_idx] = model.predict_proba(va_pool)[:, 1]
    models.append(model)

    print(f"[Fold {fold}] best_iter={model.get_best_iteration()} | best_score={model.get_best_score()}")

# 여기서 oof_pred로 threshold 튜닝(F1 최대) 하면 됨
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

# ====================
# 1) OOF 성능 확인 (AUC + threshold 스캔으로 F1 최대 찾기)
# ====================
y_true = train_df[target_col].astype(int).values

oof_auc = roc_auc_score(y_true, oof_pred)
print(f"\nOOF AUC: {oof_auc:.4f}")

best_t, best_f1 = None, -1
best_metrics = None

# threshold는 0.05~0.95 정도만 봐도 충분 (너무 극단값은 보통 의미 없음)
for t in np.linspace(0.05, 0.95, 91):
    y_hat = (oof_pred >= t).astype(int)
    f1 = f1_score(y_true, y_hat)
    if f1 > best_f1:
        best_f1 = f1
        best_t = float(t)
        best_metrics = dict(
            acc=accuracy_score(y_true, y_hat),
            precision=precision_score(y_true, y_hat, zero_division=0),
            recall=recall_score(y_true, y_hat, zero_division=0),
        )

print(
    f"Best F1={best_f1:.4f} at t={best_t:.3f} | "
    f"ACC={best_metrics['acc']:.4f} | "
    f"P={best_metrics['precision']:.4f} | "
    f"R={best_metrics['recall']:.4f}"
)

# ====================
# 2) test 예측 (폴드 모델 앙상블 평균)
#    - 각 fold에서 만든 model은 "그 fold의 스케일"로 학습됨
#    - 그래서 test도 fold마다 scaler로 변환해서 예측해야 정합성이 맞음
#    - 위 루프에서 X_te를 만들긴 했지만, 밖으로 저장을 안 했으니
#      여기서 "fold별로 다시" scaler까지 재현해서 평균내는 게 안전함
# ====================
test_pred = np.zeros(len(test_df), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(rskf.split(train_df[x_cols], train_df[target_col]), 1):
    # fold별 scaler 재현
    X_tr = train_df.loc[tr_idx, x_cols].copy()
    X_te = test_df[x_cols].copy()

    scaler = MinMaxScaler()
    X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
    X_te[num_cols] = scaler.transform(X_te[num_cols])

    te_pool = Pool(X_te, cat_features=cat_cols)

    # fold 순서대로 models에 들어있다고 가정
    model = models[fold - 1]
    test_pred += model.predict_proba(te_pool)[:, 1] / len(models)

# ====================
# 3) 최종 threshold(best_t)로 0/1 변환 & 제출 파일 생성
# ====================
test_label = (test_pred >= best_t).astype(int)

# 보통 dacon은 sample_submission.csv를 제공함
# 없으면 ID + completed로 만들면 됨
try:
    sub = pd.read_csv("/Users/parksung-cheol/Desktop/dacon-project/data/raw/sample_submission.csv")
    # 컬럼명이 completed가 아닐 수도 있으니 안전하게 처리
    if "completed" in sub.columns:
        sub["completed"] = test_label
    else:
        # 마지막 컬럼을 타겟으로 치환 (대회마다 다름)
        sub.iloc[:, -1] = test_label
except FileNotFoundError:
    sub = pd.DataFrame({"ID": test_df["ID"], "completed": test_label})

out_path = "/Users/parksung-cheol/Desktop/dacon-project/submission_catboost.csv"
sub.to_csv(out_path, index=False)

print(f"\nSaved submission: {out_path}")
print("Pred positive rate:", test_label.mean())
