"""
시계열 분할 실험: 랜덤 분할 vs 시간순 분할 비교

문제: train_test_split은 시계열 데이터에 부적합
- 미래 데이터로 과거를 예측하는 문제 발생
- 실제 성능보다 과대평가될 수 있음

실험:
- 방법 1: 랜덤 분할 (현재 방식, 문제 있음)
- 방법 2: 시간순 단순 분할
- 방법 3: 시계열 교차검증 (TimeSeriesSplit)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(path, sample_size=100000):
    """데이터 로드 및 준비"""
    print("=" * 60)
    print("Step 1: 데이터 로드 및 준비")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"전체 데이터: {len(df):,}건")

    # datetime 컬럼 변환
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 시간순 정렬 (중요!)
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    if sample_size and len(df) > sample_size:
        # 시계열이므로 랜덤 샘플링 대신 뒤쪽 데이터 사용
        df = df.tail(sample_size).reset_index(drop=True)
        print(f"샘플링 후: {len(df):,}건")
        print(f"샘플 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 피처 준비
    le_station = LabelEncoder()
    le_prev = LabelEncoder()

    df['station_encoded'] = le_station.fit_transform(df['station'].astype(str))
    df['prev_station_encoded'] = le_prev.fit_transform(df['prev_station'].astype(str))

    # 시간 순환 인코딩
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # 대체 피처 (문제 1에서 만든 것)
    df['hist_avg_by_dow_hour'] = df.groupby(
        ['station', 'prev_station', 'dayofweek', 'hour']
    )['duration'].transform('mean')

    return df


def get_features_and_target(df):
    """피처와 타겟 분리 (서비스 가능한 피처만 사용)"""

    # 서비스 가능한 피처만! (lag 피처 제외)
    features = [
        'hour', 'dayofweek', 'is_weekend', 'is_holiday',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'avg_duration', 'station_encoded', 'prev_station_encoded',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'hist_avg_by_dow_hour'  # 대체 피처
    ]

    features = [f for f in features if f in df.columns]
    target = 'delay'

    # 결측치 제거
    df_clean = df.dropna(subset=features + [target])

    return df_clean, features, target


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """모델 학습 및 평가"""
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }


def experiment_random_split(df, features, target):
    """방법 1: 랜덤 분할 (현재 방식)"""
    print("\n" + "=" * 60)
    print("방법 1: 랜덤 분할 (현재 방식)")
    print("=" * 60)
    print("⚠️  시계열 데이터에 부적합! (비교용)")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"학습 데이터: {len(X_train):,}건")
    print(f"테스트 데이터: {len(X_test):,}건")

    # 문제 시각화: 테스트 데이터의 시간 분포
    test_indices = X_test.index
    train_indices = X_train.index

    test_dates = df.loc[test_indices, 'datetime']
    train_dates = df.loc[train_indices, 'datetime']

    print(f"\n학습 데이터 기간: {train_dates.min()} ~ {train_dates.max()}")
    print(f"테스트 데이터 기간: {test_dates.min()} ~ {test_dates.max()}")
    print("→ 테스트 데이터가 학습 데이터 사이사이에 섞여있음! ❌")

    result = train_and_evaluate(X_train, X_test, y_train, y_test)

    print(f"\n📊 결과:")
    print(f"   MAE:  {result['mae']:.2f}초")
    print(f"   RMSE: {result['rmse']:.2f}초")
    print(f"   R²:   {result['r2']:.4f}")

    return result


def experiment_time_based_split(df, features, target):
    """방법 2: 시간순 단순 분할"""
    print("\n" + "=" * 60)
    print("방법 2: 시간순 단순 분할")
    print("=" * 60)
    print("✅ 올바른 방법: 과거로 미래 예측")

    # 이미 시간순 정렬되어 있음
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"학습 데이터: {len(X_train):,}건")
    print(f"테스트 데이터: {len(X_test):,}건")

    print(f"\n학습 데이터 기간: {train_df['datetime'].min()} ~ {train_df['datetime'].max()}")
    print(f"테스트 데이터 기간: {test_df['datetime'].min()} ~ {test_df['datetime'].max()}")
    print("→ 학습(과거) → 테스트(미래) 순서 ✅")

    result = train_and_evaluate(X_train, X_test, y_train, y_test)

    print(f"\n📊 결과:")
    print(f"   MAE:  {result['mae']:.2f}초")
    print(f"   RMSE: {result['rmse']:.2f}초")
    print(f"   R²:   {result['r2']:.4f}")

    return result


def experiment_timeseries_cv(df, features, target, n_splits=5):
    """방법 3: 시계열 교차검증"""
    print("\n" + "=" * 60)
    print(f"방법 3: 시계열 교차검증 ({n_splits} Folds)")
    print("=" * 60)
    print("✅ 가장 신뢰할 수 있는 방법")

    X = df[features].values
    y = df[target].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    print("\n각 Fold 결과:")
    print("-" * 50)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 기간 확인
        train_start = df.iloc[train_idx[0]]['datetime']
        train_end = df.iloc[train_idx[-1]]['datetime']
        test_start = df.iloc[test_idx[0]]['datetime']
        test_end = df.iloc[test_idx[-1]]['datetime']

        result = train_and_evaluate(X_train, X_test, y_train, y_test)
        results.append(result)

        print(f"Fold {fold}: 학습({train_start.strftime('%m/%d')}~{train_end.strftime('%m/%d')}) → "
              f"테스트({test_start.strftime('%m/%d')}~{test_end.strftime('%m/%d')}) | "
              f"MAE: {result['mae']:.2f}초")

    # 평균 계산
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_r2 = np.mean([r['r2'] for r in results])
    std_mae = np.std([r['mae'] for r in results])

    print("-" * 50)
    print(f"\n📊 평균 결과:")
    print(f"   MAE:  {avg_mae:.2f}초 (±{std_mae:.2f})")
    print(f"   RMSE: {avg_rmse:.2f}초")
    print(f"   R²:   {avg_r2:.4f}")

    return {
        'mae': avg_mae,
        'rmse': avg_rmse,
        'r2': avg_r2,
        'std_mae': std_mae,
        'fold_results': results
    }


def run_experiment(data_path):
    """실험 실행"""

    # 1. 데이터 준비
    df = load_and_prepare_data(data_path, sample_size=100000)
    df_clean, features, target = get_features_and_target(df)

    print(f"\n사용 피처: {len(features)}개")
    print(f"유효 데이터: {len(df_clean):,}건")

    # 2. 세 가지 방법 비교
    result_random = experiment_random_split(df_clean, features, target)
    result_time = experiment_time_based_split(df_clean, features, target)
    result_cv = experiment_timeseries_cv(df_clean, features, target, n_splits=5)

    # 3. 최종 비교
    print("\n" + "=" * 60)
    print("📊 최종 비교 결과")
    print("=" * 60)

    print("""
    ┌────────────────────┬─────────┬─────────┬─────────┬──────────┐
    │      분할 방법      │   MAE   │  RMSE   │   R²    │ 적합성   │
    ├────────────────────┼─────────┼─────────┼─────────┼──────────┤""")
    print(f"    │ 1. 랜덤 분할        │ {result_random['mae']:>6.2f}초 │ {result_random['rmse']:>6.2f}초 │ {result_random['r2']:>6.4f} │   ❌     │")
    print(f"    │ 2. 시간순 분할      │ {result_time['mae']:>6.2f}초 │ {result_time['rmse']:>6.2f}초 │ {result_time['r2']:>6.4f} │   ✅     │")
    print(f"    │ 3. 시계열 CV (평균) │ {result_cv['mae']:>6.2f}초 │ {result_cv['rmse']:>6.2f}초 │ {result_cv['r2']:>6.4f} │   ✅✅   │")
    print("    └────────────────────┴─────────┴─────────┴─────────┴──────────┘")

    # 4. 분석
    print("\n" + "=" * 60)
    print("💡 분석")
    print("=" * 60)

    overestimate = result_time['mae'] - result_random['mae']

    if overestimate > 0:
        print(f"""
    ⚠️  랜덤 분할이 성능을 과대평가하고 있었음!

    - 랜덤 분할 MAE: {result_random['mae']:.2f}초
    - 시간순 분할 MAE: {result_time['mae']:.2f}초
    - 차이: {overestimate:.2f}초 ({overestimate/result_time['mae']*100:.1f}% 과대평가)

    → 실제 서비스에서는 시간순 분할 기준 성능({result_time['mae']:.2f}초)이 현실적
        """)
    else:
        print(f"""
    랜덤 분할과 시간순 분할 성능이 비슷함

    - 랜덤 분할 MAE: {result_random['mae']:.2f}초
    - 시간순 분할 MAE: {result_time['mae']:.2f}초

    → 이 경우에도 시간순 분할이 더 적절한 평가 방법
        """)

    print(f"""
    📌 권장사항:
    1. 모델 평가 시 시계열 교차검증(TimeSeriesSplit) 사용
    2. 최종 성능은 {result_cv['mae']:.2f}초 (±{result_cv['std_mae']:.2f})로 보고
    3. 면접에서 "시계열 데이터 특성을 고려했다"고 설명
    """)

    return {
        'random': result_random,
        'time_based': result_time,
        'timeseries_cv': result_cv
    }


if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed_subway_data.csv'
    results = run_experiment(data_path)
