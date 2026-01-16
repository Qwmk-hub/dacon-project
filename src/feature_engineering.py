"""
피처 엔지니어링 함수
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_derived_features(df):
    """새로운 파생 변수 생성"""
    df = df.copy()
    
    # 예시: 수강한 클래스 수
    class_cols = ['class1', 'class2', 'class3', 'class4']
    if all(col in df.columns for col in class_cols):
        df['num_classes'] = df[class_cols].notna().sum(axis=1)
    
    # 예시: 이전 기수 수강 여부
    prev_class_cols = ['previous_class_3', 'previous_class_4', 'previous_class_5', 
                       'previous_class_6', 'previous_class_7', 'previous_class_8']
    if all(col in df.columns for col in prev_class_cols):
        df['prev_experience'] = df[prev_class_cols].notna().sum(axis=1)
    
    return df


def scale_features(df_train, df_test, scaler_type='standard'):
    """
    피처 스케일링
    
    Parameters:
    -----------
    scaler_type : str
        'standard': StandardScaler
        'minmax': MinMaxScaler
    """
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test_scaled[numeric_cols] = scaler.transform(df_test[numeric_cols])
    
    return df_train_scaled, df_test_scaled, scaler


def select_features(X, y, method='correlation', threshold=0.1):
    """
    피처 선택
    
    Parameters:
    -----------
    method : str
        'correlation': 상관계수 기반
        'importance': 트리 모델 기반 (추후 구현)
    """
    if method == 'correlation':
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected_features = correlations[correlations > threshold].index.tolist()
        return selected_features
    
    return X.columns.tolist()
