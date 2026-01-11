"""
서울시 지하철 지연 예측 - 데이터 전처리 파이프라인
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SubwayDataPreprocessor:
    """지하철 데이터 전처리 클래스"""

    def __init__(self, base_path: str = '/Users/vvoo/Capstone_TS'):
        self.base_path = base_path
        self.subway_data_path = os.path.join(base_path, 'data', 'subway')
        self.api_data_path = os.path.join(base_path, 'data', 'api')

        # 한국 공휴일 (2024년)
        self.holidays_2024 = [
            '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',
            '2024-03-01', '2024-04-10', '2024-05-05', '2024-05-06', '2024-05-15',
            '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18',
            '2024-10-03', '2024-10-09', '2024-12-25'
        ]

    def load_subway_duration_data(self) -> pd.DataFrame:
        """
        역간 소요시간 데이터 로딩
        Returns: 통합된 소요시간 DataFrame
        """
        all_data = []
        files = [f for f in os.listdir(self.subway_data_path) if f.endswith('.csv')]

        for file in files:
            file_path = os.path.join(self.subway_data_path, file)
            df = pd.read_csv(file_path, index_col=0)

            # 역 쌍 정보 추출 (파일명에서)
            # 형식: group_('역A', '역B').csv
            try:
                station_pair = file.replace("group_('", "").replace("').csv", "")
                station_pair = station_pair.replace("', '", ",")
                station, prev_station = station_pair.split(',')

                df['station'] = station
                df['prev_station'] = prev_station
                df['date'] = df.index
                all_data.append(df)
            except:
                continue

        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    def fill_missing_with_hourly_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치를 시간대별 평균값으로 채우기

        Args:
            df: 소요시간 데이터프레임
        Returns:
            결측치가 처리된 데이터프레임
        """
        hour_columns = [str(i) for i in range(5, 24)]

        # 역 쌍별, 요일별로 그룹화하여 평균 계산
        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek

        for col in hour_columns:
            if col in df.columns:
                # 역 쌍별, 요일별 평균 계산
                group_mean = df.groupby(['station', 'prev_station', 'dayofweek'])[col].transform('mean')

                # 결측치를 그룹 평균으로 채우기
                df[col] = df[col].fillna(group_mean)

                # 여전히 남은 결측치는 전체 평균으로
                df[col] = df[col].fillna(df[col].mean())

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 관련 피처 추가

        Args:
            df: 데이터프레임
        Returns:
            시간 피처가 추가된 데이터프레임
        """
        df['date'] = pd.to_datetime(df['date'])

        # 기본 시간 피처
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek  # 0=월요일, 6=일요일
        df['dayofweek_name'] = df['date'].dt.day_name()

        # 주말 여부
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # 공휴일 여부
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df['is_holiday'] = df['date_str'].isin(self.holidays_2024).astype(int)

        # 출퇴근 시간대 여부 (7-9시, 18-20시)
        # (이 부분은 시간대별 데이터이므로 melt 후 적용)

        return df

    def melt_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간대별 컬럼을 행으로 변환 (wide → long format)

        Args:
            df: wide format 데이터프레임
        Returns:
            long format 데이터프레임
        """
        hour_columns = [str(i) for i in range(5, 24)]
        id_vars = [col for col in df.columns if col not in hour_columns]

        melted = df.melt(
            id_vars=id_vars,
            value_vars=hour_columns,
            var_name='hour',
            value_name='duration'
        )

        melted['hour'] = melted['hour'].astype(int)

        # datetime 생성
        melted['datetime'] = pd.to_datetime(
            melted['date'].astype(str) + ' ' + melted['hour'].astype(str) + ':00:00'
        )

        return melted

    def add_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        지연 관련 피처 추가

        Args:
            df: long format 데이터프레임
        Returns:
            지연 피처가 추가된 데이터프레임
        """
        # 역 쌍별, 시간대별, 요일별 평균 소요시간
        df['avg_duration'] = df.groupby(
            ['station', 'prev_station', 'hour', 'dayofweek']
        )['duration'].transform('mean')

        # 지연 시간 (실제 - 평균)
        df['delay'] = df['duration'] - df['avg_duration']

        # 지연 비율
        df['delay_ratio'] = df['delay'] / df['avg_duration']

        # 지연 여부 (10초 이상 지연 시)
        df['is_delayed'] = (df['delay'] > 10).astype(int)

        # 심각한 지연 (30초 이상)
        df['is_severe_delay'] = (df['delay'] > 30).astype(int)

        return df

    def add_rush_hour_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        출퇴근 시간대 피처 추가

        Args:
            df: long format 데이터프레임
        Returns:
            출퇴근 피처가 추가된 데이터프레임
        """
        # 출근 시간대 (7-9시)
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)

        # 퇴근 시간대 (18-20시)
        df['is_evening_rush'] = df['hour'].isin([18, 19, 20]).astype(int)

        # 혼잡 시간대 (출퇴근 시간)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

        # 심야 시간대 (22-23시)
        df['is_late_night'] = df['hour'].isin([22, 23]).astype(int)

        return df

    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        시차(lag) 피처 추가 - 이전 시간대의 소요시간

        Args:
            df: long format 데이터프레임
            lags: lag 시점 리스트
        Returns:
            lag 피처가 추가된 데이터프레임
        """
        df = df.sort_values(['station', 'prev_station', 'datetime'])

        for lag in lags:
            df[f'duration_lag_{lag}'] = df.groupby(
                ['station', 'prev_station']
            )['duration'].shift(lag)

        # 이동 평균
        df['duration_rolling_mean_3'] = df.groupby(
            ['station', 'prev_station']
        )['duration'].transform(lambda x: x.rolling(3, min_periods=1).mean())

        return df

    def preprocess(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        전체 전처리 파이프라인 실행

        Args:
            save_path: 결과 저장 경로 (선택)
        Returns:
            전처리 완료된 데이터프레임
        """
        print("1. 데이터 로딩 중...")
        df = self.load_subway_duration_data()
        print(f"   - 로딩 완료: {len(df)} rows")

        print("2. 결측치 처리 중...")
        df = self.fill_missing_with_hourly_mean(df)

        print("3. 시간 피처 추가 중...")
        df = self.add_time_features(df)

        print("4. Long format 변환 중...")
        df = self.melt_hourly_data(df)
        print(f"   - 변환 완료: {len(df)} rows")

        print("5. 지연 피처 추가 중...")
        df = self.add_delay_features(df)

        print("6. 출퇴근 피처 추가 중...")
        df = self.add_rush_hour_features(df)

        print("7. Lag 피처 추가 중...")
        df = self.add_lag_features(df)

        # 불필요한 컬럼 정리
        df = df.drop(columns=['date_str'], errors='ignore')

        if save_path:
            print(f"8. 저장 중: {save_path}")
            df.to_csv(save_path, index=False, encoding='utf-8-sig')

        print("전처리 완료!")
        return df

    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        데이터 요약 통계

        Args:
            df: 전처리된 데이터프레임
        Returns:
            요약 통계 딕셔너리
        """
        stats = {
            'total_records': len(df),
            'unique_stations': df['station'].nunique(),
            'unique_routes': df.groupby(['station', 'prev_station']).ngroups,
            'date_range': (df['date'].min(), df['date'].max()),
            'missing_values': df.isnull().sum().to_dict(),
            'delay_stats': {
                'mean_delay': df['delay'].mean(),
                'std_delay': df['delay'].std(),
                'delayed_ratio': df['is_delayed'].mean(),
                'severe_delay_ratio': df['is_severe_delay'].mean()
            },
            'duration_stats': {
                'mean': df['duration'].mean(),
                'std': df['duration'].std(),
                'min': df['duration'].min(),
                'max': df['duration'].max()
            }
        }
        return stats


# 사용 예시
if __name__ == '__main__':
    preprocessor = SubwayDataPreprocessor()

    # 전처리 실행
    processed_df = preprocessor.preprocess(
        save_path='/Users/vvoo/Capstone_TS/data/processed_subway_data.csv'
    )

    # 요약 통계 출력
    stats = preprocessor.get_summary_stats(processed_df)
    print("\n=== 데이터 요약 ===")
    print(f"총 레코드 수: {stats['total_records']:,}")
    print(f"역 수: {stats['unique_stations']}")
    print(f"역간 구간 수: {stats['unique_routes']}")
    print(f"기간: {stats['date_range']}")
    print(f"\n=== 지연 통계 ===")
    print(f"평균 지연: {stats['delay_stats']['mean_delay']:.2f}초")
    print(f"지연 발생률: {stats['delay_stats']['delayed_ratio']*100:.2f}%")
    print(f"심각 지연률: {stats['delay_stats']['severe_delay_ratio']*100:.2f}%")
