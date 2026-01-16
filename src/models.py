"""
모델 학습 및 평가 함수
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def train_model(X_train, y_train, model_type='logistic'):
    """
    모델 학습
    
    Parameters:
    -----------
    model_type : str
        'logistic': 로지스틱 회귀
        'rf': 랜덤 포레스트
    """
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unknown model type")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """모델 평가"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba


def cross_validate(X, y, model_type='logistic', cv=5):
    """교차 검증"""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    return scores


def get_feature_importance(model):
    """피처 중요도 추출"""
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])
    else:
        return None
