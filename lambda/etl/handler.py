"""
서울시 지하철 데이터 ETL Lambda
Raw JSON → Processed Parquet 변환
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from io import BytesIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SubwayETL:
    """지하철 데이터 ETL 처리기"""

    # 한국 공휴일 (2024-2026)
    HOLIDAYS = {
        '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11',
        '2024-03-01', '2024-05-05', '2024-05-06', '2024-05-15',
        '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17',
        '2024-09-18', '2024-10-03', '2024-10-09', '2024-12-25',
        '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30',
        '2025-03-01', '2025-05-05', '2025-05-06', '2025-06-06',
        '2025-08-15', '2025-10-03', '2025-10-06', '2025-10-07',
        '2025-10-08', '2025-10-09', '2025-12-25',
        '2026-01-01'
    }

    def __init__(self, s3_client=None, bucket_name: str = None):
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    def extract_from_s3(self, date: str) -> List[Dict]:
        """S3에서 특정 날짜의 원시 데이터 추출"""
        if not self.s3_client or not self.bucket_name:
            raise ValueError("S3 client and bucket required")

        prefix = f"raw/year={date[:4]}/month={date[5:7]}/day={date[8:10]}/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
        except Exception as e:
            logger.error(f"S3 list failed: {e}")
            return []

        all_records = []
        for obj in response.get('Contents', []):
            try:
                file_obj = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
                data = json.loads(file_obj['Body'].read().decode('utf-8'))
                all_records.extend(data.get('data', []))
            except Exception as e:
                logger.warning(f"Failed to read {obj['Key']}: {e}")
                continue

        logger.info(f"Extracted {len(all_records)} records for {date}")
        return all_records

    def extract_from_local(self, data_dir: str, date: str) -> List[Dict]:
        """로컬 파일시스템에서 데이터 추출"""
        import glob

        date_dir = os.path.join(data_dir, date)
        if not os.path.exists(date_dir):
            logger.warning(f"Directory not found: {date_dir}")
            return []

        all_records = []
        for filepath in glob.glob(os.path.join(date_dir, "*.json")):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_records.extend(data.get('data', []))
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue

        logger.info(f"Extracted {len(all_records)} records from {date_dir}")
        return all_records

    def transform(self, records: List[Dict], date: str) -> List[Dict]:
        """데이터 변환 및 피처 생성"""
        if not records:
            return []

        date_obj = datetime.strptime(date, '%Y-%m-%d')
        dayofweek = date_obj.weekday()
        is_weekend = dayofweek >= 5
        is_holiday = date in self.HOLIDAYS

        # 역/시간대별 그룹화를 위한 딕셔너리
        grouped = {}

        for record in records:
            station = record.get('statnNm', '')
            prev_station = record.get('arvlMsg3', '')
            line_id = record.get('subwayId', '')
            direction = record.get('updnLine', '')

            # 도착 예정 시간 파싱
            try:
                barvlDt = int(record.get('barvlDt', 0))
            except (ValueError, TypeError):
                barvlDt = 0

            # 수신 시간에서 시간대 추출
            recptnDt = record.get('recptnDt', '')
            try:
                hour = int(recptnDt.split(' ')[1].split(':')[0])
            except:
                continue

            # 운행 시간대만 (5-23시)
            if hour < 5 or hour > 23:
                continue

            # 유효한 역/이전역만
            if not station or not prev_station:
                continue
            if prev_station in ['', ' ', '종점']:
                continue

            # 그룹 키 생성
            key = (station, prev_station, line_id, direction, hour)

            if key not in grouped:
                grouped[key] = {
                    'durations': [],
                    'station': station,
                    'prev_station': prev_station,
                    'line_id': line_id,
                    'direction': direction,
                    'hour': hour
                }

            grouped[key]['durations'].append(barvlDt)

        # 그룹별 집계
        transformed = []
        for key, group in grouped.items():
            durations = group['durations']
            if not durations:
                continue

            avg_duration = sum(durations) / len(durations)

            transformed.append({
                'date': date,
                'station': group['station'],
                'prev_station': group['prev_station'],
                'line_id': group['line_id'],
                'direction': group['direction'],
                'hour': group['hour'],
                'avg_duration': round(avg_duration, 2),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'sample_count': len(durations),
                'dayofweek': dayofweek,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'is_morning_rush': group['hour'] in [7, 8, 9],
                'is_evening_rush': group['hour'] in [18, 19, 20]
            })

        logger.info(f"Transformed to {len(transformed)} aggregated records")
        return transformed

    def load_to_s3(self, data: List[Dict], date: str) -> Dict:
        """S3에 처리된 데이터 저장"""
        if not data:
            return {'status': 'no_data'}

        # JSON Lines 형식으로 저장
        output = '\n'.join(json.dumps(row, ensure_ascii=False) for row in data)

        s3_key = f"processed/year={date[:4]}/month={date[5:7]}/day={date[8:10]}/data.jsonl"

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=output.encode('utf-8'),
            ContentType='application/json'
        )

        return {
            'status': 'success',
            's3_key': s3_key,
            'record_count': len(data)
        }

    def load_to_local(self, data: List[Dict], output_dir: str, date: str) -> Dict:
        """로컬에 처리된 데이터 저장"""
        if not data:
            return {'status': 'no_data'}

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"processed_{date}.jsonl")

        with open(output_path, 'w', encoding='utf-8') as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        return {
            'status': 'success',
            'filepath': output_path,
            'record_count': len(data)
        }

    def run(self, date: str, local_input: str = None, local_output: str = None) -> Dict:
        """ETL 파이프라인 실행"""
        logger.info(f"Starting ETL for {date}")

        # Extract
        if local_input:
            records = self.extract_from_local(local_input, date)
        else:
            records = self.extract_from_s3(date)

        if not records:
            return {'status': 'no_data', 'date': date}

        # Transform
        transformed = self.transform(records, date)

        # Load
        if local_output:
            result = self.load_to_local(transformed, local_output, date)
        else:
            result = self.load_to_s3(transformed, date)

        result['date'] = date
        return result


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda 핸들러

    이벤트 형식:
    {
        "date": "2024-05-01"  # 처리할 날짜 (없으면 어제)
    }
    """
    import boto3

    # 환경변수
    bucket_name = os.environ.get('S3_BUCKET')
    if not bucket_name:
        raise ValueError("S3_BUCKET environment variable required")

    # 처리할 날짜 결정 (기본: 어제)
    date = event.get('date')
    if not date:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime('%Y-%m-%d')

    # ETL 실행
    s3_client = boto3.client('s3')
    etl = SubwayETL(s3_client=s3_client, bucket_name=bucket_name)

    try:
        result = etl.run(date)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error(f"ETL failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# 로컬 실행용
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='지하철 데이터 ETL')
    parser.add_argument('--date', required=True, help='처리할 날짜 (YYYY-MM-DD)')
    parser.add_argument('--input-dir', required=True, help='입력 데이터 디렉토리')
    parser.add_argument('--output-dir', default='./data/processed', help='출력 디렉토리')

    args = parser.parse_args()

    etl = SubwayETL()
    result = etl.run(
        date=args.date,
        local_input=args.input_dir,
        local_output=args.output_dir
    )

    print(f"ETL 결과: {json.dumps(result, indent=2)}")
