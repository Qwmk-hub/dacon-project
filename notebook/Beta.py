from __future__ import annotations
import pandas as pd
import importlib
import Alpha
importlib.reload(Alpha)
from Alpha import merge_df, completed

import numpy as np
import optuna

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# --------------------
# Load
# --------------------
train_raw = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\train.csv")
test_raw  = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\test.csv")

train_df = merge_df(train_raw)
test_df  = merge_df(test_raw)
train_df = completed(train_raw, train_df)

train_df = train_df.sort_values("ID").reset_index(drop=True)
test_df  = test_df.sort_values("ID").reset_index(drop=True)

total_df1 = train_df.drop(columns=["ID", "completed"])
total_df2 = test_df.drop(columns=["ID"])

train_data = total_df1
test_data = total_df2

target = train_df["completed"]

cat_cols = [
    "school1","job","nationality","High Tech","Data Friendly","Others",
    "hope_for_group","desired_career_path",
    "incumbents_level","incumbents_lecture","incumbents_company_level",
    "incumbents_lecture_type","incumbents_lecture_scale",
]
num_cols = ["count","time_input","want_count"]

for col in num_cols:
    q3 = train_data[col].quantile(0.75)
    q1 = train_data[col].quantile(0.25)
    iqr = q3 - q1
    outlier_idx = train_data[col][(train_data[col]<q1 - 1.5*iqr)|(train_data[col]>q3 + 1.5*iqr)].index
    train_data.drop(outlier_idx, inplace=True)
    target.drop(outlier_idx, inplace=True)

column_names_to_normalize = num_cols
x = train_data[column_names_to_normalize].values
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
train_data_temp = pd.DataFrame(x_scaled, columns = column_names_to_normalize, index = train_data.index)
train_data[column_names_to_normalize] = train_data_temp

column_names_to_normalize = num_cols
x = test_data[column_names_to_normalize].values

test_scaled =  scaler.transform(x)
test_temp = pd.DataFrame(test_scaled, columns = column_names_to_normalize, index = test_data.index)
test_data[column_names_to_normalize] = test_temp

X_train, X_validation, y_train, y_validation = train_test_split(train_data, target, train_size = 0.7, random_state = 42)

train_pool  = Pool(X_train, y_train, cat_features=cat_cols)
eval_pool = Pool(X_validation, y_validation, cat_features=cat_cols)
test_pool = Pool(data = test_data, cat_features=cat_cols)

sampler = TPESampler(seed = 10)

# 함수 정의
def objective(trial):

    param = {
      "random_state" : 42,
      'learning_rate' : trial.suggest_uniform('learning_rate', 0.01, 0.2),
    } 
    model = CatBoostClassifier(**param)
    f1_list = []
    kf = KFold(n_splits=10)
    for tr_index,val_index in kf.split(train_data):
        X_train, y_train = train_data.iloc[tr_index], target.iloc[tr_index]
        X_valid , y_valid = train_data.iloc[val_index], target.iloc[val_index]
        model = model.fit(X_train,y_train, eval_set=[(X_train,y_train),(X_valid,y_valid)],
                           verbose=False, early_stopping_rounds=35)                         
        f1_list.append(f1_score(y_valid, model.predict(X_valid),average='macro'))
    return np.mean(f1_list)

optuna_cbrm = optuna.create_study(direction="maximize", sampler=sampler)
optuna_cbrm.optimize(objective, n_trials = 30)

cbrm_trial = optuna_cbrm.best_trial
cbrm_trial_params = cbrm_trial.params

cbrm_trial_params

#Optuna에서 가져온 최적의 파라미터들로 모델 학습
params = {
          'learning_rate': 0.1550157115572994,
          'eval_metric':'AUC',
          'early_stopping_rounds':50,
          'use_best_model': True,
          'random_seed': 42,
          'auto_class_weights':'Balanced',
          'verbose':200}
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=eval_pool,use_best_model=True)

pred = model.predict(eval_pool)
print(classification_report(y_validation,pred,digits=5))

# ----------------제출용-----------------------------
mypredictions = model.predict(test_data)

ss = pd.read_csv('C:\\Users\\solba\\dacon-project\\result\\sample_submission.csv',header=0)
ss['completed'] = mypredictions

ss.to_csv('My_submission.csv',index=False)