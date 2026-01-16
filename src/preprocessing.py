"""
데이터 전처리 함수
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(df, strategy='drop'):
    """
    결측값 처리
    
    Parameters:
    -----------
    df : pd.DataFrame
    strategy : str
        'drop': 결측값이 있는 행 삭제
        'mean': 숫자 컬럼은 평균, 범주형은 최빈값으로 채우기
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def encode_categorical(df, categorical_cols=None):
    """범주형 변수 인코딩"""
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    return df, le_dict


def remove_outliers(df, numeric_cols=None, method='iqr'):
    """
    이상치 제거
    
    Parameters:
    -----------
    method : str
        'iqr': IQR 방법
        'zscore': Z-score 방법
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'iqr':
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                 (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        mask = (z_scores < 3).all(axis=1)
    
    return df[mask]
