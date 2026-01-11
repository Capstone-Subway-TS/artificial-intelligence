"""
피처 실험: Lag 피처 vs 대체 피처 비교
- 모델 A: 원래 모델 (lag 포함) - 서비스 불가하지만 기준점
- 모델 B: lag 제거 - 서비스 가능, 성능 하락 예상
- 모델 C: 대체 피처 - 서비스 가능, 성능 회복 목표
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(path, sample_size=100000):
    """데이터 로드"""
    print("=" * 60)
    print("Step 1: 데이터 로드")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"전체 데이터: {len(df):,}건")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"샘플링 후: {len(df):,}건")

    return df


def prepare_base_features(df):
    """기본 피처 준비 (인코딩 등)"""
    df = df.copy()

    # 범주형 인코딩
    le_station = LabelEncoder()
    le_prev = LabelEncoder()

    df['station_encoded'] = le_station.fit_transform(df['station'].astype(str))
    df['prev_station_encoded'] = le_prev.fit_transform(df['prev_station'].astype(str))

    # 시간 순환 인코딩
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df


def create_alternative_features(df):
    """
    대체 피처 생성: 예측 시점에 알 수 있는 정보만 사용

    핵심 아이디어:
    - lag 피처는 "지금 상황"을 반영 (실시간, 하지만 예측 시점에 모름)
    - 대체 피처는 "과거 같은 상황"을 반영 (통계, 예측 시점에 알 수 있음)
    """
    df = df.copy()

    print("\n대체 피처 생성 중...")

    # 1. 같은 역, 같은 요일, 같은 시간대의 과거 평균
    # → "지난 월요일 8시에 이 구간은 평균 몇 초 걸렸나?"
    df['hist_avg_by_dow_hour'] = df.groupby(
        ['station', 'prev_station', 'dayofweek', 'hour']
    )['duration'].transform('mean')

    # 2. 같은 역, 같은 시간대의 과거 평균 (요일 무관)
    # → "8시에 이 구간은 보통 몇 초 걸리나?"
    df['hist_avg_by_hour'] = df.groupby(
        ['station', 'prev_station', 'hour']
    )['duration'].transform('mean')

    # 3. 같은 역, 같은 요일의 과거 평균 (시간 무관)
    # → "월요일에 이 구간은 보통 몇 초 걸리나?"
    df['hist_avg_by_dow'] = df.groupby(
        ['station', 'prev_station', 'dayofweek']
    )['duration'].transform('mean')

    # 4. 같은 역의 전체 평균 대비 현재 시간대 비율
    # → "이 시간대는 평소보다 얼마나 더 걸리나?"
    df['hour_vs_avg_ratio'] = df['hist_avg_by_hour'] / (df['avg_duration'] + 1)

    # 5. 출퇴근 강도 점수
    rush_intensity = {
        7: 0.7, 8: 1.0, 9: 0.8,  # 오전 러시
        18: 0.8, 19: 1.0, 20: 0.7  # 오후 러시
    }
    df['rush_intensity'] = df['hour'].map(rush_intensity).fillna(0)

    # 6. 주말/공휴일 + 시간대 조합
    df['weekend_hour_interaction'] = df['is_weekend'] * df['hour']

    print("  - hist_avg_by_dow_hour: 요일+시간대별 과거 평균")
    print("  - hist_avg_by_hour: 시간대별 과거 평균")
    print("  - hist_avg_by_dow: 요일별 과거 평균")
    print("  - hour_vs_avg_ratio: 시간대별 혼잡 비율")
    print("  - rush_intensity: 출퇴근 강도 점수")

    return df


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    """모델 학습 및 평가"""
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'model': model
    }


def get_feature_importance(model, feature_names, top_n=10):
    """피처 중요도 출력"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance.head(top_n)


def run_experiment(data_path):
    """실험 실행"""

    # 1. 데이터 로드
    df = load_data(data_path, sample_size=100000)

    # 2. 기본 피처 준비
    df = prepare_base_features(df)

    # 3. 대체 피처 생성
    df = create_alternative_features(df)

    # 타겟 변수
    target = 'delay'

    # 결측치 제거
    df = df.dropna(subset=[target])

    # ========================================
    # 모델 A: 원래 피처 (lag 포함)
    # ========================================
    print("\n" + "=" * 60)
    print("모델 A: 원래 모델 (lag 피처 포함)")
    print("=" * 60)
    print("⚠️  이 모델은 실제 서비스에서 사용 불가! (비교용)")

    features_a = [
        'hour', 'dayofweek', 'is_weekend', 'is_holiday',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'avg_duration', 'station_encoded', 'prev_station_encoded',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        # lag 피처들 (예측 시점에 알 수 없음!)
        'duration_lag_1', 'duration_lag_2', 'duration_lag_3',
        'duration_rolling_mean_3'
    ]

    # 피처 존재 여부 확인
    features_a = [f for f in features_a if f in df.columns]
    print(f"사용 피처: {len(features_a)}개")

    # 결측치 있는 행 제거
    df_a = df.dropna(subset=features_a + [target])
    print(f"유효 데이터: {len(df_a):,}건")

    X_a = df_a[features_a]
    y_a = df_a[target]

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_a, y_a, test_size=0.2, random_state=42
    )

    result_a = train_and_evaluate(X_train_a, X_test_a, y_train_a, y_test_a, "모델 A (lag 포함)")
    print(f"\n📊 결과:")
    print(f"   MAE:  {result_a['mae']:.2f}초")
    print(f"   RMSE: {result_a['rmse']:.2f}초")
    print(f"   R²:   {result_a['r2']:.4f}")

    print("\n📈 피처 중요도 (Top 5):")
    imp_a = get_feature_importance(result_a['model'], features_a, 5)
    for _, row in imp_a.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:25s} {bar} {row['importance']*100:.1f}%")

    # ========================================
    # 모델 B: lag 피처 제거
    # ========================================
    print("\n" + "=" * 60)
    print("모델 B: lag 피처 제거")
    print("=" * 60)
    print("✅ 이 모델은 실제 서비스에서 사용 가능!")

    features_b = [
        'hour', 'dayofweek', 'is_weekend', 'is_holiday',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'avg_duration', 'station_encoded', 'prev_station_encoded',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        # lag 피처 제거됨!
    ]

    features_b = [f for f in features_b if f in df.columns]
    print(f"사용 피처: {len(features_b)}개")

    df_b = df.dropna(subset=features_b + [target])
    print(f"유효 데이터: {len(df_b):,}건")

    X_b = df_b[features_b]
    y_b = df_b[target]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y_b, test_size=0.2, random_state=42
    )

    result_b = train_and_evaluate(X_train_b, X_test_b, y_train_b, y_test_b, "모델 B (lag 제거)")
    print(f"\n📊 결과:")
    print(f"   MAE:  {result_b['mae']:.2f}초")
    print(f"   RMSE: {result_b['rmse']:.2f}초")
    print(f"   R²:   {result_b['r2']:.4f}")

    # 성능 변화
    mae_diff_b = result_b['mae'] - result_a['mae']
    print(f"\n   vs 모델A: MAE {mae_diff_b:+.2f}초 ({'악화' if mae_diff_b > 0 else '개선'})")

    print("\n📈 피처 중요도 (Top 5):")
    imp_b = get_feature_importance(result_b['model'], features_b, 5)
    for _, row in imp_b.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:25s} {bar} {row['importance']*100:.1f}%")

    # ========================================
    # 모델 C: 대체 피처 추가
    # ========================================
    print("\n" + "=" * 60)
    print("모델 C: 대체 피처 추가")
    print("=" * 60)
    print("✅ 이 모델은 실제 서비스에서 사용 가능!")

    features_c = [
        # 기본 피처
        'hour', 'dayofweek', 'is_weekend', 'is_holiday',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'avg_duration', 'station_encoded', 'prev_station_encoded',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        # 대체 피처 (예측 시점에 알 수 있음!)
        'hist_avg_by_dow_hour',   # 요일+시간대별 과거 평균
        'hist_avg_by_hour',       # 시간대별 과거 평균
        'hist_avg_by_dow',        # 요일별 과거 평균
        'hour_vs_avg_ratio',      # 시간대별 혼잡 비율
        'rush_intensity',         # 출퇴근 강도
        'weekend_hour_interaction' # 주말x시간 상호작용
    ]

    features_c = [f for f in features_c if f in df.columns]
    print(f"사용 피처: {len(features_c)}개")

    df_c = df.dropna(subset=features_c + [target])
    print(f"유효 데이터: {len(df_c):,}건")

    X_c = df_c[features_c]
    y_c = df_c[target]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42
    )

    result_c = train_and_evaluate(X_train_c, X_test_c, y_train_c, y_test_c, "모델 C (대체 피처)")
    print(f"\n📊 결과:")
    print(f"   MAE:  {result_c['mae']:.2f}초")
    print(f"   RMSE: {result_c['rmse']:.2f}초")
    print(f"   R²:   {result_c['r2']:.4f}")

    # 성능 변화
    mae_diff_c_vs_a = result_c['mae'] - result_a['mae']
    mae_diff_c_vs_b = result_c['mae'] - result_b['mae']
    print(f"\n   vs 모델A: MAE {mae_diff_c_vs_a:+.2f}초")
    print(f"   vs 모델B: MAE {mae_diff_c_vs_b:+.2f}초 ({'개선!' if mae_diff_c_vs_b < 0 else '악화'})")

    print("\n📈 피처 중요도 (Top 10):")
    imp_c = get_feature_importance(result_c['model'], features_c, 10)
    for _, row in imp_c.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:25s} {bar} {row['importance']*100:.1f}%")

    # ========================================
    # 최종 비교
    # ========================================
    print("\n" + "=" * 60)
    print("📊 최종 비교 결과")
    print("=" * 60)

    print("""
    ┌──────────────────┬─────────┬─────────┬─────────┬──────────┐
    │      모델        │   MAE   │  RMSE   │   R²    │ 서비스   │
    ├──────────────────┼─────────┼─────────┼─────────┼──────────┤""")
    print(f"    │ A. 원래 (lag포함) │ {result_a['mae']:>6.2f}초 │ {result_a['rmse']:>6.2f}초 │ {result_a['r2']:>6.4f} │   ❌     │")
    print(f"    │ B. lag 제거       │ {result_b['mae']:>6.2f}초 │ {result_b['rmse']:>6.2f}초 │ {result_b['r2']:>6.4f} │   ✅     │")
    print(f"    │ C. 대체 피처      │ {result_c['mae']:>6.2f}초 │ {result_c['rmse']:>6.2f}초 │ {result_c['r2']:>6.4f} │   ✅     │")
    print("    └──────────────────┴─────────┴─────────┴─────────┴──────────┘")

    # 결론
    print("\n" + "=" * 60)
    print("💡 결론")
    print("=" * 60)

    if result_c['mae'] < result_b['mae']:
        recovery = (result_b['mae'] - result_c['mae']) / (result_b['mae'] - result_a['mae']) * 100
        print(f"""
    ✅ 대체 피처가 효과적!

    - 모델 B (lag 제거) 대비 MAE {result_b['mae'] - result_c['mae']:.2f}초 개선
    - lag 제거로 인한 성능 하락의 {recovery:.1f}% 회복
    - 실제 서비스에서 사용 가능한 모델 C 권장
        """)
    else:
        print(f"""
    ⚠️ 대체 피처 효과 미미

    - 추가 피처 엔지니어링 필요
    - 또는 다른 모델 시도 필요 (시계열 모델 등)
        """)

    return {
        'model_a': result_a,
        'model_b': result_b,
        'model_c': result_c
    }


if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed_subway_data.csv'
    results = run_experiment(data_path)
