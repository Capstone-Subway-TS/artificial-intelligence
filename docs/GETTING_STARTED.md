# 시작 가이드

## 1. 사전 요구사항

### 필수 도구
- Python 3.11+
- Docker & Docker Compose
- AWS CLI (배포 시)
- AWS SAM CLI (배포 시)

### API 키 발급
1. [서울 열린데이터 광장](https://data.seoul.go.kr/) 회원가입
2. 마이페이지 > 인증키 발급 신청
3. "서울시 지하철 실시간 도착정보" API 사용 신청

---

## 2. 로컬 개발 환경 설정

### 2.1 프로젝트 클론
```bash
git clone https://github.com/your-repo/subway-delay-prediction.git
cd subway-delay-prediction
```

### 2.2 Python 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r lambda/collector/requirements.txt
pip install pytest  # 테스트용
```

### 2.3 환경변수 설정
```bash
# .env 파일 생성
cat > .env << EOF
SEOUL_API_KEY=your_api_key_here
EOF

# 환경변수 로드
export $(cat .env | xargs)
```

---

## 3. 로컬 테스트

### 3.1 단위 테스트 실행
```bash
cd tests
pytest test_collector.py -v
```

### 3.2 수집기 로컬 실행 (단일 실행)
```bash
cd lambda/collector
python handler.py --api-key $SEOUL_API_KEY --output-dir ../../data/collected
```

### 3.3 연속 수집 모드
```bash
python handler.py --api-key $SEOUL_API_KEY --output-dir ../../data/collected --continuous
# Ctrl+C로 종료
```

### 3.4 Docker로 실행 (LocalStack 포함)
```bash
# LocalStack + 수집기 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f collector

# 종료
docker-compose down
```

---

## 4. AWS 배포 (프리티어 범위)

### 4.1 AWS CLI 설정
```bash
aws configure
# AWS Access Key ID: your_access_key
# AWS Secret Access Key: your_secret_key
# Default region name: ap-northeast-2
# Default output format: json
```

### 4.2 SAM CLI 설치
```bash
# macOS
brew install aws-sam-cli

# Linux
pip install aws-sam-cli

# 설치 확인
sam --version
```

### 4.3 배포 실행
```bash
cd infrastructure

# 환경변수 설정
export SEOUL_API_KEY=your_api_key_here

# 개발 환경 배포
./deploy.sh dev

# 운영 환경 배포 (필요시)
./deploy.sh prod
```

### 4.4 배포 확인
```bash
# Lambda 함수 확인
aws lambda list-functions --query 'Functions[?starts_with(FunctionName, `subway-`)].FunctionName'

# S3 버킷 확인
aws s3 ls | grep subway

# 최근 로그 확인
aws logs tail /aws/lambda/subway-collector-dev --since 1h
```

---

## 5. 비용 모니터링

### 5.1 예상 비용 (프리티어)
| 서비스 | 월 사용량 | 프리티어 한도 | 비용 |
|--------|----------|--------------|------|
| Lambda | ~21,600건 | 100만 건 | $0 |
| S3 | ~500MB | 5GB | $0 |
| CloudWatch | ~100MB | 5GB | $0 |

### 5.2 비용 알람 설정 (권장)
```bash
# AWS Budgets 설정 (월 $1 초과 시 알림)
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget file://budget.json \
    --notifications-with-subscribers file://notifications.json
```

---

## 6. 문제 해결

### API 호출 실패
```bash
# API 키 테스트
curl "http://swopenAPI.seoul.go.kr/api/subway/${SEOUL_API_KEY}/json/realtimeStationArrival/1/5/강남"
```

### Lambda 오류 확인
```bash
# 최근 오류 로그
aws logs filter-log-events \
    --log-group-name /aws/lambda/subway-collector-dev \
    --filter-pattern "ERROR"
```

### S3 데이터 확인
```bash
# 최근 수집 데이터 목록
aws s3 ls s3://subway-delay-prediction-dev-xxx/raw/ --recursive | tail -10

# 샘플 데이터 다운로드
aws s3 cp s3://subway-delay-prediction-dev-xxx/raw/year=2024/month=05/day=01/train_arrival_10-30-00.json ./sample.json
```

---

## 7. 다음 단계

1. **Glue ETL 구축**: 데이터 전처리 파이프라인
2. **ML 모델 개발**: 지연 예측 모델
3. **API 개발**: FastAPI 백엔드
4. **대시보드**: 프론트엔드 개발
