"""
ML 모델 - 라우터 통합 모듈

기존 ML 모델을 라우터와 연동하여 구간별 지연 예측
lag 피처는 과거 평균값으로 대체하여 실시간 예측 가능하게 함
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 상위 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLDelayPredictor:
    """ML 기반 지연 예측기"""

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: 학습된 모델 경로 (.joblib)
        """
        if model_path is None:
            # 기본 경로 (여러 위치 시도)
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'models'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ml', 'models'),
                '/Users/vvoo/Capstone_TS/ml/models',
            ]

            for model_dir in possible_paths:
                if os.path.exists(model_dir):
                    models = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
                    if models:
                        model_path = os.path.join(model_dir, sorted(models)[-1])
                        break

        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_type = None

        # 과거 평균값 (lag 피처 대체용)
        self.historical_stats = {
            'duration_lag_1': 120,  # 평균 2분
            'duration_lag_2': 120,
            'duration_lag_3': 120,
            'duration_rolling_mean_3': 120,
            'avg_duration': 120
        }

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"모델을 찾을 수 없습니다: {model_path}")

    def _load_model(self, model_path: str):
        """모델 로드"""
        try:
            data = joblib.load(model_path)
            self.model = data.get('model')
            self.label_encoders = data.get('label_encoders', {})
            self.feature_columns = data.get('feature_columns', [])
            self.model_type = data.get('model_type', 'unknown')
            print(f"모델 로드 완료: {os.path.basename(model_path)}")
            print(f"  타입: {self.model_type}")
            print(f"  피처 수: {len(self.feature_columns)}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")

    def _encode_station(self, station: str, encoder_name: str) -> int:
        """역명 인코딩"""
        encoder = self.label_encoders.get(encoder_name)
        if encoder is None:
            return 0

        try:
            if station in encoder.classes_:
                return int(encoder.transform([station])[0])
            else:
                return -1  # 미지의 역
        except:
            return 0

    def predict_delay(
        self,
        from_station: str,
        to_station: str,
        line: int,
        departure_time: datetime = None
    ) -> float:
        """
        단일 구간 지연 예측

        Args:
            from_station: 출발역
            to_station: 도착역
            line: 호선
            departure_time: 출발 시간

        Returns:
            예측 지연시간 (초)
        """
        if self.model is None:
            return 0.0

        if departure_time is None:
            departure_time = datetime.now()

        hour = departure_time.hour
        dayofweek = departure_time.weekday()
        is_weekend = int(dayofweek >= 5)

        # 피처 구성
        features = {
            'hour': hour,
            'dayofweek': dayofweek,
            'is_weekend': is_weekend,
            'is_holiday': 0,  # 공휴일 정보 없음
            'is_morning_rush': int(7 <= hour <= 9),
            'is_evening_rush': int(18 <= hour <= 20),
            'is_rush_hour': int((7 <= hour <= 9) or (18 <= hour <= 20)),

            # lag 피처 (과거 평균으로 대체)
            'duration_lag_1': self.historical_stats['duration_lag_1'],
            'duration_lag_2': self.historical_stats['duration_lag_2'],
            'duration_lag_3': self.historical_stats['duration_lag_3'],
            'duration_rolling_mean_3': self.historical_stats['duration_rolling_mean_3'],
            'avg_duration': self.historical_stats['avg_duration'],

            # 순환 피처
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * dayofweek / 7),
            'dow_cos': np.cos(2 * np.pi * dayofweek / 7),

            # 역 인코딩
            'station_encoded': self._encode_station(to_station, 'station'),
            'prev_station_encoded': self._encode_station(from_station, 'prev_station'),
        }

        # DataFrame 생성
        df = pd.DataFrame([features])

        # 필요한 컬럼만 선택
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_columns]

        # 예측
        try:
            pred = self.model.predict(X)[0]
            return max(0, float(pred))  # 음수 방지
        except Exception as e:
            print(f"예측 오류: {e}")
            return 0.0

    def predict_route_delays(
        self,
        segments: List[Tuple[str, str, int]],
        departure_time: datetime = None
    ) -> Dict[Tuple[str, str, int], float]:
        """
        경로의 모든 구간 지연 예측

        Args:
            segments: [(출발역, 도착역, 호선), ...]
            departure_time: 출발 시간

        Returns:
            {(출발역, 도착역, 호선): 예측지연(초)}
        """
        predictions = {}

        for from_station, to_station, line in segments:
            delay = self.predict_delay(from_station, to_station, line, departure_time)
            predictions[(from_station, to_station, line)] = delay

        return predictions


def test_integration():
    """ML 모델 - 라우터 통합 테스트"""
    print("=" * 60)
    print("ML 모델 - 라우터 통합 테스트")
    print("=" * 60)

    # 1. ML 예측기 초기화
    print("\n[1] ML 예측기 초기화")
    predictor = MLDelayPredictor()

    if predictor.model is None:
        print("모델이 없어서 테스트를 진행할 수 없습니다.")
        return

    # 2. 단일 구간 예측 테스트
    print("\n[2] 단일 구간 지연 예측")
    test_segments = [
        ("강남", "역삼", 2),
        ("서울역", "시청", 1),
        ("신도림", "대림", 2),
    ]

    dt = datetime(2024, 1, 15, 8, 30)  # 출근 시간
    print(f"    시간: {dt.strftime('%Y-%m-%d %H:%M')} (출근)")

    for from_st, to_st, line in test_segments:
        delay = predictor.predict_delay(from_st, to_st, line, dt)
        print(f"    {from_st} → {to_st} ({line}호선): {delay:.1f}초 지연")

    # 3. 시간대별 비교
    print("\n[3] 시간대별 지연 예측 (강남→역삼)")
    for hour in [8, 12, 18, 22]:
        dt = datetime(2024, 1, 15, hour, 0)
        delay = predictor.predict_delay("강남", "역삼", 2, dt)
        time_label = {8: "출근", 12: "평시", 18: "퇴근", 22: "심야"}[hour]
        print(f"    {hour}시 ({time_label}): {delay:.1f}초")

    # 4. 라우터와 연동
    print("\n[4] 라우터 연동 테스트")
    try:
        from subpick_router_v2 import SubpickRouterV2

        # ODsay API 키 확인
        odsay_key = os.environ.get('ODSAY_API_KEY')

        router = SubpickRouterV2()

        # ML 예측값을 라우터에 설정
        # 주요 구간들의 지연 예측
        segments = [
            ("강남", "역삼", 2), ("역삼", "선릉", 2), ("선릉", "삼성", 2),
            ("서울역", "시청", 1), ("시청", "종각", 1),
            ("신도림", "대림", 2), ("대림", "구로디지털단지", 2),
        ]

        dt = datetime(2024, 1, 15, 18, 30)  # 퇴근 시간
        delays = predictor.predict_route_delays(segments, dt)

        print(f"    예측된 지연 구간 수: {len(delays)}")
        print(f"    평균 지연: {sum(delays.values()) / len(delays):.1f}초")

        # 라우터에 지연 설정
        router.router.set_predicted_delays(delays)

        # 경로 추천
        print("\n[5] 지연 반영 경로 추천 (서울역 → 강남)")
        result = router.recommend_route("서울역", "강남", "2024-01-15 18:30")

        print(f"\n    추천 결과:")
        print(f"      총 소요: {result.primary_route.total_time}초")
        print(f"      환승: {result.primary_route.transfer_count}회")
        print(f"      환승 실패 위험: {result.primary_route.transfer_failure_risk:.1%}")

    except ImportError as e:
        print(f"    라우터 import 실패: {e}")
    except Exception as e:
        print(f"    라우터 테스트 실패: {e}")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == '__main__':
    test_integration()
