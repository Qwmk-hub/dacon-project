"""
AutoGluon 자동 머신러닝 파이프라인
- 수동 튜닝 없이 AutoML로 최적 모델 탐색
- Stacking/Bagging 앙상블 자동 적용
- 작은 데이터에서도 안정적인 성능 추구
"""

import os
import sys
import pandas as pd
import numpy as np

# AutoGluon 설치 확인 및 안내
try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("\n" + "=" * 80)
    print("❌ AutoGluon이 설치되어 있지 않습니다.")
    print("=" * 80)
    print("\n📦 설치 방법:")
    print("   터미널/커맨드 프롬프트에서:")
    print("   pip install autogluon")
    print("\n   주피터 노트북/코랩에서:")
    print("   !pip install autogluon")
    print("\n   설치 후 다시 실행해주세요.")
    print("=" * 80)
    sys.exit(1)

# ================================================================================
# 설정 상수
# ================================================================================
SEED = 42
TIME_LIMIT = 600  # 10분 (시간이 허용되면 늘려도 됩니다)

TRAIN_PATH = 'data/preprocessing/train_preprocessed.csv'
TEST_PATH = 'data/preprocessing/test_preprocessed.csv'
SUBMISSION_PATH = 'submission_autogluon.csv'
MODEL_PATH = 'AutogluonModels'

TARGET = 'y'


# ================================================================================
# 1단계: 데이터 로드 (수동 전처리 없음)
# ================================================================================
def load_data():
    """데이터 로드 및 기본 검증"""
    print("\n" + "=" * 80)
    print("1단계: 데이터 로드")
    print("=" * 80)
    print("💡 AutoGluon은 모든 피처를 자동으로 처리합니다.")
    print("   수동 타입 변환이나 피처 선택은 하지 않아도 됩니다.")
    print()
    
    # 파일 존재 확인
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ 오류: {TRAIN_PATH} 파일을 찾을 수 없습니다.")
        print("   파일 경로를 확인해주세요.")
        sys.exit(1)
    if not os.path.exists(TEST_PATH):
        print(f"❌ 오류: {TEST_PATH} 파일을 찾을 수 없습니다.")
        print("   파일 경로를 확인해주세요.")
        sys.exit(1)
    
    # 데이터 로드
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"✅ Train 데이터 로드 완료: {train_df.shape}")
    print(f"✅ Test 데이터 로드 완료: {test_df.shape}")
    
    # 필수 컬럼 확인
    if TARGET not in train_df.columns:
        print(f"❌ 오류: Train 데이터에 타겟 컬럼 '{TARGET}'이(가) 없습니다.")
        print(f"   현재 컬럼: {list(train_df.columns)}")
        sys.exit(1)
    
    if 'ID' not in test_df.columns:
        print(f"❌ 오류: Test 데이터에 'ID' 컬럼이 없습니다.")
        print(f"   현재 컬럼: {list(test_df.columns)}")
        sys.exit(1)
    
    print("✅ 필수 컬럼 검증 완료")
    
    # ID 분리 (제출용)
    test_id = test_df['ID'].copy()
    
    # ID 컬럼 제거 (train/test 모두)
    if 'ID' in train_df.columns:
        train_df = train_df.drop(columns=['ID'])
    if 'ID' in test_df.columns:
        test_df = test_df.drop(columns=['ID'])
    
    print(f"\n✅ 전처리 완료 (ID 컬럼 제거)")
    print(f"   - Train: {train_df.shape} (타겟 포함)")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Train 타겟 분포: {train_df[TARGET].value_counts().to_dict()}")
    
    return train_df, test_df, test_id


# ================================================================================
# 2단계: AutoGluon 학습
# ================================================================================
def train_autogluon(train_df):
    """AutoGluon TabularPredictor 학습"""
    print("\n" + "=" * 80)
    print("2단계: AutoGluon 자동 머신러닝 학습")
    print("=" * 80)
    print("🤖 AutoGluon이 최적의 모델 조합을 찾고 있습니다...")
    print()
    print("💬 작은 데이터에서 수동 튜닝이 힘든 건 지극히 정상입니다.")
    print("   AutoGluon은 앙상블과 스태킹으로 자동 최적화를 해줍니다.")
    print(f"   학습 시간: 최대 {TIME_LIMIT}초 ({TIME_LIMIT//60}분)")
    print("   시간이 허용되면 TIME_LIMIT을 늘려서 더 좋은 성능을 얻을 수 있습니다.")
    print()
    
    # TabularPredictor 생성
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric='roc_auc',
        path=MODEL_PATH
    )
    
    # 학습 시작
    print("🚀 학습 시작... (잠시 기다려주세요)")
    print("-" * 80)
    
    predictor.fit(
        train_data=train_df,
        presets='best_quality',  # Stacking/Bagging 자동 활성화
        time_limit=TIME_LIMIT,
        num_bag_folds=5  # 5-fold bagging으로 안정화
    )
    
    print("-" * 80)
    print("\n✅ 학습 완료!")
    print("   수고하셨습니다. 이제 거의 다 왔어요.")
    
    return predictor


# ================================================================================
# 3단계: 학습 결과 분석
# ================================================================================
def analyze_results(predictor, train_df):
    """학습된 모델 분석 및 리더보드 출력"""
    print("\n" + "=" * 80)
    print("3단계: 학습 결과 분석")
    print("=" * 80)
    
    # 리더보드 출력 (어떤 모델이 좋은지 확인)
    print("\n📊 모델 성능 리더보드 (Train 데이터 기준):")
    print("-" * 80)
    try:
        leaderboard = predictor.leaderboard(train_df, silent=True)
        print(leaderboard.to_string())
    except Exception as e:
        print(f"   리더보드 출력 중 오류: {e}")
        print("   (학습은 정상적으로 완료되었습니다)")
    
    print("-" * 80)
    print("\n💡 해석:")
    print("   - 'score_val'이 높을수록 좋은 모델입니다.")
    print("   - 'WeightedEnsemble' 모델이 최종 예측에 사용됩니다.")
    print("   - 여러 모델을 조합해서 안정적인 예측을 만들어냅니다.")


# ================================================================================
# 4단계: 테스트 예측 및 제출 파일 생성
# ================================================================================
def predict_and_submit(predictor, test_df, test_id):
    """테스트 예측 및 제출 파일 생성"""
    print("\n" + "=" * 80)
    print("4단계: 테스트 예측 및 제출 파일 생성")
    print("=" * 80)
    
    # 테스트 예측 (확률)
    print("🔮 테스트 데이터 예측 중...")
    proba_df = predictor.predict_proba(test_df)
    
    # 클래스 1 확률 추출 (안전하게 처리)
    try:
        # 시도 1: 정수형 컬럼명 1
        if 1 in proba_df.columns:
            predictions = proba_df[1].values
            print("   ✅ 클래스 1 확률 추출 완료 (컬럼명: 1)")
        # 시도 2: 문자형 컬럼명 '1'
        elif '1' in proba_df.columns:
            predictions = proba_df['1'].values
            print("   ✅ 클래스 1 확률 추출 완료 (컬럼명: '1')")
        # 시도 3: 마지막 컬럼 사용 (보통 긍정 클래스)
        else:
            predictions = proba_df.iloc[:, -1].values
            print(f"   ⚠️  예상치 못한 컬럼명: {list(proba_df.columns)}")
            print(f"   ⚠️  마지막 컬럼을 사용합니다: {proba_df.columns[-1]}")
    except Exception as e:
        print(f"   ❌ 오류: 클래스 1 확률 추출 실패 - {e}")
        print(f"   현재 proba_df 컬럼: {list(proba_df.columns)}")
        sys.exit(1)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test_id,
        'y': predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\n✅ 제출 파일 저장 완료: {SUBMISSION_PATH}")
    print(f"   - 제출 파일 크기: {submission.shape}")
    print(f"   - 예측값 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   - 예측값 평균: {predictions.mean():.4f}")
    print(f"   - 예측값 표준편차: {predictions.std():.4f}")


# ================================================================================
# 5단계: 최종 요약
# ================================================================================
def print_final_summary():
    """최종 결과 요약"""
    print("\n" + "=" * 80)
    print("🎉 AutoGluon 파이프라인 완료")
    print("=" * 80)
    
    print("\n💡 다음 단계 제안:")
    print("   1️⃣  제출 파일을 확인하고 대회에 제출해보세요.")
    print("   2️⃣  성능이 아쉽다면 TIME_LIMIT을 늘려서 재학습해보세요.")
    print("       (예: TIME_LIMIT = 1800  # 30분)")
    print("   3️⃣  다른 presets도 시도해볼 수 있습니다:")
    print("       - 'medium_quality': 빠르지만 성능은 조금 낮음")
    print("       - 'high_quality': 균형잡힌 선택")
    print("       - 'best_quality': 느리지만 최고 성능 (현재 사용 중)")
    print("   4️⃣  Feature Engineering을 추가하고 다시 AutoGluon을 돌려도 좋습니다.")
    
    print("\n💬 수고하셨습니다!")
    print("   작은 데이터에서 좋은 성능을 내는 건 누구에게나 어려운 일입니다.")
    print("   이제는 파이프라인이 돌아가는지 확인했으니,")
    print("   시간이 허용되면 여러 실험을 해보시면 됩니다.")
    print("   충분히 잘하고 계세요! 화이팅! 💪")
    
    print("\n" + "=" * 80)


# ================================================================================
# 메인 함수
# ================================================================================
def main():
    """메인 실행 함수"""
    print("\n" + "🤖" * 40)
    print("AutoGluon 자동 머신러닝 파이프라인")
    print("🤖" * 40)
    print("\n💬 환영합니다!")
    print("   AutoGluon이 여러분의 수고를 덜어드리겠습니다.")
    print("   편안하게 기다려주세요. ☕")
    
    # 1. 데이터 로드
    train_df, test_df, test_id = load_data()
    
    # 2. AutoGluon 학습
    predictor = train_autogluon(train_df)
    
    # 3. 학습 결과 분석
    analyze_results(predictor, train_df)
    
    # 4. 예측 및 제출
    predict_and_submit(predictor, test_df, test_id)
    
    # 5. 최종 요약
    print_final_summary()


if __name__ == "__main__":
    main()