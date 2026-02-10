import pandas as pd
import numpy as np
import re

CURRENT_CLASS_COLS = ["class1", "class2", "class3", "class4"]
PREV_CLASS_COLS = [f"previous_class_{i}" for i in range(3, 9)]
CLASS_COLS = CURRENT_CLASS_COLS + PREV_CLASS_COLS

STOP_TOKENS = {"없음", "없습니다", "아직 없음", "미정", "해당없음", "none", "n/a", "na", "nan"}

def _is_taken(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    s = str(x).strip()
    if not s:
        return False
    low = s.lower()
    if low in STOP_TOKENS:
        return False
    if ("없" in low) or ("미정" in low):
        return False
    return True

def _split_certs(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = re.split(r"[,/;]+", s)
    out = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        low = t.lower()
        if low in STOP_TOKENS:
            continue
        if ("없" in low) or ("미정" in low):
            continue
        out.append(t)
    return out

def _cert_count_row(row):
    owned = _split_certs(row.get("certificate_acquisition"))
    desired = _split_certs(row.get("desired_certificate"))
    return len(set(owned) | set(desired))  # 중복 제거

def num(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # 1) class_count 생성 (여기가 빠져서 에러났던 거)
    df["class_count"] = df[CLASS_COLS].applymap(_is_taken).sum(axis=1)

    # 2) certificate_count 생성
    df["certificate_count"] = df.apply(_cert_count_row, axis=1)

    # 3) time_input 숫자화
    df["time_input"] = pd.to_numeric(df["time_input"], errors="coerce")

    # 4) 필요한 컬럼만 반환 (test엔 completed 없을 수 있음)
    cols = ["ID", "time_input", "class_count", "certificate_count"]
    if "completed" in df.columns:
        cols.append("completed")

    return df[cols]

def cat(data):
    data = data.copy()
    data_cat = data[['ID', 'major_data', 're_registration', 'incumbents_level']]
    return data_cat
