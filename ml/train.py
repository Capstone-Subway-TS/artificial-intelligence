"""
서울시 지하철 지연 예측 모델 학습
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# LightGBM import (없으면 sklearn 사용)
HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    # 테스트 임포트
    lgb.LGBMRegressor()
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor


class DelayPredictionModel:
    """지연 예측 모델"""

    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type if HAS_LIGHTGBM else 'sklearn'
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링"""
        df = df.copy()

        # 기본 피처 확인 및 생성
        if 'hour' not in df.columns and 'datetime' in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour

        if 'dayofweek' not in df.columns and 'datetime' in df.columns:
            df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek

        # 결측치 처리
        df['duration'] = df['duration'].fillna(df['duration'].mean())

        # 범주형 변수 인코딩
        categorical_cols = ['station', 'prev_station']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    # 학습 시 없던 값 처리
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_
                        else -1
                    )

        # 시간 순환 피처 (sin/cos)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'dayofweek' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """모델 학습에 사용할 피처 컬럼 선택"""
        feature_cols = []

        # 숫자형 피처
        numeric_features = [
            'hour', 'dayofweek', 'is_weekend', 'is_holiday',
            'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
            'duration_lag_1', 'duration_lag_2', 'duration_lag_3',
            'duration_rolling_mean_3', 'avg_duration',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'station_encoded', 'prev_station_encoded'
        ]

        for col in numeric_features:
            if col in df.columns:
                feature_cols.append(col)

        return feature_cols

    def train(self, df: pd.DataFrame, target_col: str = 'delay') -> Dict[str, float]:
        """모델 학습"""
        print("1. 피처 준비 중...")
        df = self.prepare_features(df)

        # 피처/타겟 분리
        self.feature_columns = self.get_feature_columns(df)
        print(f"   사용 피처: {len(self.feature_columns)}개")

        # 유효한 데이터만 선택
        valid_mask = df[target_col].notna() & df[self.feature_columns].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        print(f"   유효 데이터: {len(df_valid)}개 / {len(df)}개")

        X = df_valid[self.feature_columns]
        y = df_valid[target_col]

        # Train/Test 분리
        print("2. 데이터 분할 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 모델 학습
        print("3. 모델 학습 중...")
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X_train, y_train)

        # 예측 및 평가
        print("4. 모델 평가 중...")
        y_pred = self.model.predict(X_test)

        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(self.feature_columns)
        }

        print(f"\n=== 모델 성능 ===")
        print(f"MAE (평균 절대 오차): {self.metrics['mae']:.2f}초")
        print(f"RMSE (평균 제곱근 오차): {self.metrics['rmse']:.2f}초")
        print(f"R² Score: {self.metrics['r2']:.4f}")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("Model not trained")

        df = self.prepare_features(df)
        X = df[self.feature_columns]
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """피처 중요도"""
        if self.model is None:
            raise ValueError("Model not trained")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, model_dir: str) -> str:
        """모델 저장"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 모델 저장
        model_path = os.path.join(model_dir, f'model_{timestamp}.joblib')
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, model_path)

        # 메타데이터 저장
        meta_path = os.path.join(model_dir, f'metadata_{timestamp}.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'model_type': self.model_type,
                'metrics': self.metrics,
                'feature_columns': self.feature_columns
            }, f, indent=2)

        print(f"\n모델 저장: {model_path}")
        return model_path

    @classmethod
    def load(cls, model_path: str) -> 'DelayPredictionModel':
        """모델 로드"""
        data = joblib.load(model_path)

        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.label_encoders = data['label_encoders']
        instance.feature_columns = data['feature_columns']

        return instance


def main():
    """메인 학습 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='지연 예측 모델 학습')
    parser.add_argument('--data', required=True, help='학습 데이터 경로 (CSV)')
    parser.add_argument('--output', default='./models', help='모델 저장 경로')
    parser.add_argument('--target', default='delay', help='타겟 컬럼')

    args = parser.parse_args()

    # 데이터 로드
    print(f"데이터 로드: {args.data}")
    df = pd.read_csv(args.data)
    print(f"데이터 크기: {df.shape}")

    # 모델 학습
    model = DelayPredictionModel()
    metrics = model.train(df, target_col=args.target)

    # 피처 중요도 출력
    importance = model.get_feature_importance()
    if not importance.empty:
        print("\n=== 피처 중요도 (Top 10) ===")
        print(importance.head(10).to_string(index=False))

    # 모델 저장
    model.save(args.output)

    return metrics


if __name__ == '__main__':
    main()
