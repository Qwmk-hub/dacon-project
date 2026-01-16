"""
프로젝트 설정
"""

# 경로 설정
DATA_DIR = 'data'
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# 파일명
TRAIN_FILE = 'data/raw/train.csv'
TEST_FILE = 'data/raw/test.csv'
SUBMISSION_FILE = 'data/raw/sample_submission.csv'

# 전처리 설정
MISSING_VALUE_STRATEGY = 'mean'  # 'drop' or 'mean'
OUTLIER_REMOVAL = True
OUTLIER_METHOD = 'iqr'  # 'iqr' or 'zscore'

# 모델 설정
MODEL_TYPE = 'rf'  # 'logistic' or 'rf'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 스케일링
SCALING = True
SCALER_TYPE = 'standard'  # 'standard' or 'minmax'

# 피처 엔지니어링
CREATE_DERIVED_FEATURES = True
FEATURE_SELECTION = True
CORRELATION_THRESHOLD = 0.05
