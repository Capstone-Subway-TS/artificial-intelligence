# Seoul Subway Real-time Delay Prediction Service

서울시 지하철 실시간 지연 예측 서비스 - 종합 대시보드

## Overview

서울 지하철 1~8호선의 실시간 도착 정보를 수집하여 지연 시간을 예측하고, 사용자에게 최적의 경로를 제안하는 서비스입니다.

### Key Features
- **Real-time Monitoring**: 2분 간격 실시간 데이터 수집
- **Delay Prediction**: ML 기반 지연 시간 예측
- **Route Optimization**: 지연 상황 반영 최적 경로 추천
- **Dashboard**: 종합 모니터링 대시보드

## Architecture

```
[Seoul Metro API] → [Lambda] → [S3] → [Glue ETL] → [Athena/ML] → [API] → [Dashboard]
```

### Tech Stack
| Layer | Technology |
|-------|------------|
| Data Collection | AWS Lambda, EventBridge |
| Data Lake | AWS S3 |
| Data Processing | AWS Glue, Spark |
| ML/Analytics | SageMaker, Athena |
| Backend API | FastAPI |
| Frontend | React (planned) |
| IaC | AWS SAM, CloudFormation |

## Project Structure

```
.
├── lambda/
│   └── collector/          # 데이터 수집 Lambda
│       ├── handler.py
│       ├── requirements.txt
│       └── Dockerfile
├── Model/
│   ├── data_processing/    # 전처리 파이프라인
│   ├── subway_prediction/  # 예측 모델
│   └── subway_route/       # 경로 탐색
├── infrastructure/
│   ├── template.yaml       # SAM 템플릿
│   └── deploy.sh          # 배포 스크립트
├── docs/
│   ├── architecture/       # 아키텍처 문서
│   └── GETTING_STARTED.md  # 시작 가이드
├── tests/                  # 테스트 코드
├── data/                   # 수집 데이터 (gitignore)
└── docker-compose.yml      # 로컬 개발 환경
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker
- AWS CLI & SAM CLI
- Seoul Open API Key ([발급 링크](https://data.seoul.go.kr/))

### Local Development
```bash
# 1. 의존성 설치
pip install -r lambda/collector/requirements.txt

# 2. 환경변수 설정
export SEOUL_API_KEY=your_api_key

# 3. 로컬 실행
cd lambda/collector
python handler.py --api-key $SEOUL_API_KEY --output-dir ../../data/collected
```

### AWS Deployment
```bash
cd infrastructure
export SEOUL_API_KEY=your_api_key
./deploy.sh dev
```

자세한 내용은 [시작 가이드](docs/GETTING_STARTED.md)를 참조하세요.

## Data Pipeline

### Data Sources
| Source | Update Frequency | Purpose |
|--------|-----------------|---------|
| 실시간 도착정보 API | 2분 | 열차 위치, 도착 예정 |
| 역간 소요시간 | 일 1회 집계 | 지연 계산 기준 |

### Data Flow
```
Raw Data (JSON) → Processed (Parquet) → Features → ML Model → Predictions
```

## ML Model

### Delay Definition
> 지연 = 실제 소요시간 - 해당 시간대 평균 소요시간

### Features
- 시간 피처: hour, dayofweek, is_holiday, is_rush_hour
- 지연 피처: duration_lag_1~3, rolling_mean
- 혼잡도 피처: 승하차 인원 (planned)

### Model Selection
- LightGBM (primary)
- LSTM (time series)
- Prophet (seasonality)

## Cost Optimization

AWS 프리티어 범위 내 운영:

| Service | Monthly Usage | Free Tier | Cost |
|---------|--------------|-----------|------|
| Lambda | ~21,600 calls | 1M calls | $0 |
| S3 | ~500MB | 5GB | $0 |
| CloudWatch | ~100MB | 5GB | $0 |

## Documentation

- [System Architecture](docs/architecture/system_architecture.md)
- [Data Pipeline Design](docs/architecture/data_pipeline.md)
- [Getting Started](docs/GETTING_STARTED.md)

## Roadmap

- [x] Phase 1: Data Collection Pipeline
- [ ] Phase 2: ETL & Data Warehouse
- [ ] Phase 3: ML Model Development
- [ ] Phase 4: API & Dashboard
- [ ] Phase 5: Production Deployment

## License

This project is for educational and portfolio purposes.

## Contact

- GitHub: [@your-username](https://github.com/your-username)
