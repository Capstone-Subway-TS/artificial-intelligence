"""
서울시 지하철 실시간 도착정보 수집 Lambda
AWS Lambda 및 로컬 환경 모두에서 실행 가능
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import requests

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SubwayCollector:
    """서울시 지하철 API 수집기"""

    def __init__(self, api_key: str, s3_client=None, bucket_name: str = None):
        self.api_key = api_key
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.api_url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimeStationArrival/ALL"

    def fetch_data(self) -> Dict[str, Any]:
        """API에서 데이터 수집"""
        try:
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API 호출 실패: {e}")
            raise

    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 변환 및 메타데이터 추가"""
        now = datetime.now()
        arrival_list = raw_data.get('realtimeArrivalList', [])

        # 필요한 필드만 추출
        cleaned_records = []
        for record in arrival_list:
            cleaned_records.append({
                'subwayId': record.get('subwayId'),
                'statnNm': record.get('statnNm'),
                'statnId': record.get('statnId'),
                'updnLine': record.get('updnLine'),
                'trainLineNm': record.get('trainLineNm'),
                'barvlDt': record.get('barvlDt'),
                'btrainNo': record.get('btrainNo'),
                'bstatnNm': record.get('bstatnNm'),
                'arvlMsg2': record.get('arvlMsg2'),
                'arvlMsg3': record.get('arvlMsg3'),
                'arvlCd': record.get('arvlCd'),
                'recptnDt': record.get('recptnDt')
            })

        return {
            'collection_time': now.isoformat(),
            'source': 'seoul_metro_api',
            'record_count': len(cleaned_records),
            'data': cleaned_records
        }

    def generate_s3_key(self) -> str:
        """S3 저장 경로 생성"""
        now = datetime.now()
        return (
            f"raw/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
            f"train_arrival_{now.strftime('%H-%M-%S')}.json"
        )

    def save_to_s3(self, data: Dict[str, Any], s3_key: str) -> Dict[str, Any]:
        """S3에 데이터 저장"""
        if not self.s3_client or not self.bucket_name:
            raise ValueError("S3 client and bucket name are required")

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=json.dumps(data, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )

        return {
            'bucket': self.bucket_name,
            's3_key': s3_key,
            'record_count': data['record_count']
        }

    def save_to_local(self, data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """로컬 파일시스템에 데이터 저장 (테스트용)"""
        now = datetime.now()
        date_dir = os.path.join(output_dir, now.strftime('%Y-%m-%d'))
        os.makedirs(date_dir, exist_ok=True)

        filename = f"train_arrival_{now.strftime('%H-%M-%S')}.json"
        filepath = os.path.join(date_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {
            'filepath': filepath,
            'record_count': data['record_count']
        }

    def collect(self, save_local: bool = False, local_output_dir: str = None) -> Dict[str, Any]:
        """전체 수집 프로세스 실행"""
        # 1. 데이터 수집
        logger.info("Fetching data from Seoul Metro API...")
        raw_data = self.fetch_data()

        # 2. 데이터 변환
        logger.info("Transforming data...")
        transformed_data = self.transform_data(raw_data)
        logger.info(f"Collected {transformed_data['record_count']} records")

        # 3. 저장
        if save_local:
            result = self.save_to_local(transformed_data, local_output_dir)
            logger.info(f"Saved to local: {result['filepath']}")
        else:
            s3_key = self.generate_s3_key()
            result = self.save_to_s3(transformed_data, s3_key)
            logger.info(f"Saved to S3: s3://{result['bucket']}/{result['s3_key']}")

        return result


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda 핸들러

    환경변수:
        - SEOUL_API_KEY: 서울시 오픈API 키
        - S3_BUCKET: S3 버킷 이름
    """
    import boto3

    # 환경변수 읽기
    api_key = os.environ.get('SEOUL_API_KEY')
    bucket_name = os.environ.get('S3_BUCKET')

    if not api_key:
        raise ValueError("SEOUL_API_KEY environment variable is required")
    if not bucket_name:
        raise ValueError("S3_BUCKET environment variable is required")

    # S3 클라이언트 생성
    s3_client = boto3.client('s3')

    # 수집기 실행
    collector = SubwayCollector(
        api_key=api_key,
        s3_client=s3_client,
        bucket_name=bucket_name
    )

    try:
        result = collector.collect()
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data collected successfully',
                **result
            })
        }
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


# 로컬 실행용
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='서울시 지하철 데이터 수집기')
    parser.add_argument('--api-key', required=True, help='서울시 오픈API 키')
    parser.add_argument('--output-dir', default='./data/collected',
                        help='출력 디렉토리 (기본: ./data/collected)')
    parser.add_argument('--continuous', action='store_true',
                        help='연속 수집 모드 (2분 간격)')

    args = parser.parse_args()

    collector = SubwayCollector(api_key=args.api_key)

    if args.continuous:
        import time
        print("연속 수집 모드 시작 (Ctrl+C로 종료)")
        while True:
            try:
                result = collector.collect(
                    save_local=True,
                    local_output_dir=args.output_dir
                )
                print(f"[{datetime.now()}] 수집 완료: {result['record_count']}건")
                time.sleep(120)  # 2분 대기
            except KeyboardInterrupt:
                print("\n수집 종료")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(60)
    else:
        result = collector.collect(
            save_local=True,
            local_output_dir=args.output_dir
        )
        print(f"수집 완료: {result}")
