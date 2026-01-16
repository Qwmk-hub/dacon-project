# BDA 수료 예측 프로젝트 파일 구조

```
dacon/
├── data/
│   ├── raw/
│   │   ├── train.csv           # 원본 학습 데이터
│   │   ├── test.csv            # 원본 테스트 데이터
│   │   └── sample_submission.csv # 제출 양식
│   └── processed/
│       ├── train_processed.csv  # 전처리된 학습 데이터
│       └── test_processed.csv   # 전처리된 테스트 데이터
│
├── notebooks/
│   ├── 01_EDA.ipynb             # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb    # 데이터 전처리
│   ├── 03_feature_engineering.ipynb # 피처 엔지니어링
│   ├── 04_modeling.ipynb         # 모델 학습 및 검증
│   └── 05_submission.ipynb       # 최종 제출 파일 생성
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # 전처리 함수
│   ├── feature_engineering.py    # 피처 엔지니어링 함수
│   ├── models.py                 # 모델 클래스 및 학습 함수
│   └── utils.py                  # 유틸리티 함수
│
├── models/
│   ├── model_v1.pkl             # 저장된 모델
│   └── scaler.pkl               # 저장된 스케일러
│
├── results/
│   ├── submission_v1.csv         # 제출 파일
│   ├── predictions.csv           # 예측 결과
│   └── metrics.json              # 모델 성능 지표
│
├── README.md                      # 프로젝트 설명
├── requirements.txt               # 필요 라이브러리
└── config.py                      # 설정 파일
```

## 파일별 설명

### Notebooks (순서대로 실행)
1. **01_EDA.ipynb**: 데이터 이해
   - 기본 통계
   - 결측값 분석
   - 클래스 분포 확인
   - 칼럼별 분포 시각화

2. **02_preprocessing.ipynb**: 데이터 정제
   - 결측값 처리
   - 이상치 처리
   - 데이터 타입 변환
   - 범주형 변수 인코딩

3. **03_feature_engineering.ipynb**: 새로운 특성 생성
   - 파생 변수 생성
   - 특성 스케일링
   - 특성 선택

4. **04_modeling.ipynb**: 모델 개발
   - 다양한 모델 학습 (로지스틱 회귀, 랜덤포레스트, XGBoost 등)
   - 교차 검증
   - 하이퍼파라미터 튜닝
   - 모델 비교

5. **05_submission.ipynb**: 최종 제출
   - 최적 모델 선택
   - 테스트 데이터 예측
   - 제출 파일 생성

### Source Code (src/)
- **preprocessing.py**: 학습/테스트 데이터에 공통으로 적용할 전처리 함수
- **feature_engineering.py**: 피처 엔지니어링 함수
- **models.py**: 모델 학습, 검증, 예측 함수
- **utils.py**: 데이터 로드, 저장 등 유틸리티

### 기타
- **config.py**: 임계값, 모델 파라미터 등 설정
- **requirements.txt**: pandas, scikit-learn, xgboost, lightgbm 등
- **README.md**: 프로젝트 개요 및 실행 방법
