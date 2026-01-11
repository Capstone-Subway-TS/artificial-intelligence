"""
서울시 지하철 지연 예측 모델 - 개선 버전
하이퍼파라미터 튜닝 및 모델 비교
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ImprovedDelayModel:
    """개선된 지연 예측 모델"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.results = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링 (개선)"""
        df = df.copy()

        # 기존 피처 준비
        if 'hour' not in df.columns and 'datetime' in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        if 'dayofweek' not in df.columns and 'datetime' in df.columns:
            df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek

        # 결측치 처리
        df['duration'] = df['duration'].fillna(df['duration'].mean())

        # 범주형 인코딩
        for col in ['station', 'prev_station']:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_ else -1
                    )

        # 순환 피처
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'dayofweek' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 새로운 피처
        # 1. 이전 소요시간 변화율
        if 'duration_lag_1' in df.columns and 'duration_lag_2' in df.columns:
            df['duration_change'] = df['duration_lag_1'] - df['duration_lag_2']

        # 2. 평균 대비 비율
        if 'duration_lag_1' in df.columns and 'avg_duration' in df.columns:
            df['duration_ratio'] = df['duration_lag_1'] / (df['avg_duration'] + 1)

        # 3. 시간대별 혼잡 점수
        rush_hours = {7: 0.8, 8: 1.0, 9: 0.9, 18: 0.9, 19: 1.0, 20: 0.8}
        if 'hour' in df.columns:
            df['rush_score'] = df['hour'].map(rush_hours).fillna(0)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """피처 컬럼 선택"""
        feature_cols = []
        candidates = [
            'hour', 'dayofweek', 'is_weekend', 'is_holiday',
            'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
            'duration_lag_1', 'duration_lag_2', 'duration_lag_3',
            'duration_rolling_mean_3', 'avg_duration',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'station_encoded', 'prev_station_encoded',
            'duration_change', 'duration_ratio', 'rush_score'
        ]
        for col in candidates:
            if col in df.columns:
                feature_cols.append(col)
        return feature_cols

    def train_and_compare(self, df: pd.DataFrame, target_col: str = 'delay') -> Dict:
        """여러 모델 학습 및 비교"""
        print("=" * 50)
        print("모델 학습 및 비교")
        print("=" * 50)

        # 피처 준비
        print("\n1. 피처 준비 중...")
        df = self.prepare_features(df)
        self.feature_columns = self.get_feature_columns(df)
        print(f"   사용 피처: {len(self.feature_columns)}개")

        # 유효 데이터
        valid_mask = df[target_col].notna() & df[self.feature_columns].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        print(f"   유효 데이터: {len(df_valid):,}개")

        X = df_valid[self.feature_columns]
        y = df_valid[target_col]

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # 모델 정의
        model_configs = {
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }
        }

        # 각 모델 학습
        best_score = float('inf')

        for name, config in model_configs.items():
            print(f"\n2. {name} 학습 중...")

            # GridSearchCV
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            # 최적 모델로 예측
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # 평가
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            self.results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'best_params': grid_search.best_params_
            }
            self.models[name] = best_model

            print(f"   MAE: {mae:.2f}초")
            print(f"   RMSE: {rmse:.2f}초")
            print(f"   R²: {r2:.4f}")
            print(f"   Best Params: {grid_search.best_params_}")

            # 최고 모델 추적
            if mae < best_score:
                best_score = mae
                self.best_model = best_model
                self.best_model_name = name

        # 결과 요약
        print("\n" + "=" * 50)
        print("모델 비교 결과")
        print("=" * 50)
        print(f"\n최고 성능 모델: {self.best_model_name}")
        print(f"MAE: {self.results[self.best_model_name]['mae']:.2f}초")

        return self.results

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.best_model is None:
            raise ValueError("Model not trained")

        df = self.prepare_features(df)
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)

    def save(self, model_dir: str) -> str:
        """모델 저장"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_path = os.path.join(model_dir, f'improved_model_{timestamp}.joblib')
        joblib.dump({
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'all_models': self.models,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'results': self.results
        }, model_path)

        # 메타데이터
        meta_path = os.path.join(model_dir, f'improved_metadata_{timestamp}.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_model': self.best_model_name,
                'results': {k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv
                              for kk, vv in v.items()} for k, v in self.results.items()},
                'feature_columns': self.feature_columns
            }, f, indent=2)

        print(f"\n모델 저장: {model_path}")
        return model_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='개선된 지연 예측 모델 학습')
    parser.add_argument('--data', required=True, help='학습 데이터 경로')
    parser.add_argument('--output', default='./models', help='모델 저장 경로')
    parser.add_argument('--sample', type=int, default=None, help='샘플 크기 (빠른 테스트용)')

    args = parser.parse_args()

    # 데이터 로드
    print(f"데이터 로드: {args.data}")
    df = pd.read_csv(args.data)

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"샘플링: {len(df):,}개")

    print(f"데이터 크기: {df.shape}")

    # 모델 학습
    model = ImprovedDelayModel()
    results = model.train_and_compare(df)

    # 저장
    model.save(args.output)

    return results


if __name__ == '__main__':
    main()
