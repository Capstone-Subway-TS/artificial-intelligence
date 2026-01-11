"""
수집기 단위 테스트
"""

import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda', 'collector'))

from handler import SubwayCollector, lambda_handler


class TestSubwayCollector:
    """SubwayCollector 클래스 테스트"""

    @pytest.fixture
    def collector(self):
        """테스트용 수집기 인스턴스"""
        return SubwayCollector(api_key='test_api_key')

    @pytest.fixture
    def mock_api_response(self):
        """모의 API 응답"""
        return {
            'realtimeArrivalList': [
                {
                    'subwayId': 1002,
                    'statnNm': '강남',
                    'statnId': '1002000222',
                    'updnLine': '상행',
                    'trainLineNm': '역삼행',
                    'barvlDt': 120,
                    'btrainNo': '2345',
                    'bstatnNm': '성수',
                    'arvlMsg2': '역삼 도착',
                    'arvlMsg3': '역삼',
                    'arvlCd': 1,
                    'recptnDt': '2024-05-01 10:30:00'
                },
                {
                    'subwayId': 1002,
                    'statnNm': '역삼',
                    'statnId': '1002000223',
                    'updnLine': '상행',
                    'trainLineNm': '강남행',
                    'barvlDt': 90,
                    'btrainNo': '2346',
                    'bstatnNm': '성수',
                    'arvlMsg2': '선릉 출발',
                    'arvlMsg3': '선릉',
                    'arvlCd': 2,
                    'recptnDt': '2024-05-01 10:30:00'
                }
            ]
        }

    def test_transform_data(self, collector, mock_api_response):
        """데이터 변환 테스트"""
        result = collector.transform_data(mock_api_response)

        assert 'collection_time' in result
        assert result['source'] == 'seoul_metro_api'
        assert result['record_count'] == 2
        assert len(result['data']) == 2
        assert result['data'][0]['statnNm'] == '강남'
        assert result['data'][1]['barvlDt'] == 90

    def test_transform_data_empty(self, collector):
        """빈 데이터 변환 테스트"""
        result = collector.transform_data({'realtimeArrivalList': []})

        assert result['record_count'] == 0
        assert result['data'] == []

    def test_generate_s3_key(self, collector):
        """S3 키 생성 테스트"""
        s3_key = collector.generate_s3_key()

        assert s3_key.startswith('raw/year=')
        assert 'month=' in s3_key
        assert 'day=' in s3_key
        assert s3_key.endswith('.json')

    @patch('handler.requests.get')
    def test_fetch_data_success(self, mock_get, collector, mock_api_response):
        """API 호출 성공 테스트"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response

        result = collector.fetch_data()

        assert result == mock_api_response
        mock_get.assert_called_once()

    @patch('handler.requests.get')
    def test_fetch_data_failure(self, mock_get, collector):
        """API 호출 실패 테스트"""
        mock_get.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as exc_info:
            collector.fetch_data()

        assert "Connection error" in str(exc_info.value)

    def test_save_to_local(self, collector, mock_api_response, tmp_path):
        """로컬 저장 테스트"""
        transformed = collector.transform_data(mock_api_response)
        result = collector.save_to_local(transformed, str(tmp_path))

        assert 'filepath' in result
        assert result['record_count'] == 2
        assert os.path.exists(result['filepath'])

        # 저장된 파일 내용 확인
        with open(result['filepath'], 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        assert saved_data['record_count'] == 2


class TestLambdaHandler:
    """Lambda 핸들러 테스트"""

    @patch.dict(os.environ, {
        'SEOUL_API_KEY': 'test_key',
        'S3_BUCKET': 'test-bucket'
    })
    @patch('handler.boto3.client')
    @patch('handler.requests.get')
    def test_lambda_handler_success(self, mock_get, mock_boto3):
        """Lambda 핸들러 성공 테스트"""
        # Mock API 응답
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'realtimeArrivalList': [
                {'subwayId': 1002, 'statnNm': '강남', 'barvlDt': 120}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock S3 클라이언트
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3

        result = lambda_handler({}, None)

        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['message'] == 'Data collected successfully'

    @patch.dict(os.environ, {}, clear=True)
    def test_lambda_handler_missing_env(self):
        """환경변수 누락 테스트"""
        with pytest.raises(ValueError) as exc_info:
            lambda_handler({}, None)

        assert "SEOUL_API_KEY" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
