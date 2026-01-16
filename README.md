# BDA 수료 예측 프로젝트

## 프로젝트 개요
데이콘의 BDA (Big Data Association) 학회 수료 예측 대회

### 주요 정보
- **대회명**: BDA 학회원 수료 예측
- **기간**: 2026.01.12 ~ 2026.02.23
- **참여자**: 397명
- **D-day**: 38일
- **목표**: 수료 여부 (0: 미수료, 1: 수료) 예측
- **평가지표**: TBD

## 데이터셋 설명
- **학습 데이터**: train.csv (750 rows × 45 columns)
- **테스트 데이터**: test.csv
- **타겟변수**: completed (0/1 이진 분류)

### 주요 피처
1. 기본 정보: ID, generation, school1, nationality
2. 전공 정보: major type, major1_1, major1_2, major_data, major_field
3. 학회 참여: class1~4, re_registration, contest_award
4. 경력 관련: job, desired_career_path, completed_semester
5. 직무 희망: desired_job, desired_certificate, certificate_acquisition
6. 현직자 강연: incumbents_level, incumbents_lecture, etc.
7. 관심사: interested_company, expected_domain, contest_participation

## 프로젝트 구조
```
dacon/
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 데이터
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_submission.ipynb
├── src/
│   ├── utils.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── models.py
├── models/                     # 학습된 모델 저장
├── results/                    # 결과 저장
├── config.py                   # 프로젝트 설정
└── requirements.txt            # 필요 라이브러리
```

## 실행 순서

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 노트북 순서대로 실행
1. **01_EDA.ipynb**: 탐색적 데이터 분석
2. **02_preprocessing.ipynb**: 데이터 전처리
3. **03_feature_engineering.ipynb**: 피처 엔지니어링
4. **04_modeling.ipynb**: 모델 학습 및 검증
5. **05_submission.ipynb**: 제출 파일 생성

## 분석 계획

### Phase 1: 데이터 이해 (2-3일)
- 결측값 분석
- 클래스 불균형 확인
- 피처별 분포 분석
- 상관관계 분석

### Phase 2: 전처리 (2-3일)
- 결측값 처리
- 이상치 제거
- 범주형 변수 인코딩
- 데이터 스케일링

### Phase 3: 피처 엔지니어링 (3-4일)
- 파생 변수 생성
- 피처 선택
- 다중공선성 확인

### Phase 4: 모델링 (5-7일)
- 여러 모델 학습 (로지스틱, RF, XGBoost, LightGBM)
- 하이퍼파라미터 튜닝
- 앙상블 모델 구축

### Phase 5: 제출 (2-3일)
- 최적 모델 선택
- 테스트 데이터 예측
- 제출 파일 생성

## 참고사항
- 클래스 불균형 대처: SMOTE, 클래스 가중치 등
- 크로스 검증 활용으로 과적합 방지
- 리더보드 점수 기반 모델 선택
