"""
유틸리티 함수 모음
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path


def load_data(filepath):
    """CSV 파일 로드"""
    return pd.read_csv(filepath)


def save_data(df, filepath):
    """DataFrame을 CSV로 저장"""
    df.to_csv(filepath, index=False)


def save_model(model, filepath):
    """모델을 pickle로 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """pickle 모델 로드"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_metrics(metrics, filepath):
    """성능 지표를 JSON으로 저장"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(filepath):
    """JSON 메트릭 로드"""
    with open(filepath, 'r') as f:
        return json.load(f)


def display_info(df):
    """데이터프레임 기본 정보 출력"""
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
