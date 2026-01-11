# 서울시 지하철 실시간 지연 예측 서비스 - 시스템 아키텍처

## 1. 프로젝트 개요

### 1.1 목표
일반 시민을 위한 **서울시 지하철 실시간 지연 예측 종합 대시보드** 개발

### 1.2 핵심 기능
- 실시간 지연 현황 모니터링
- 지연 시간 예측 (ML 기반)
- 지연 상황 반영 최적 경로 추천
- 알림 서비스 (Push/Email)

### 1.3 기술 스택 요약
| 영역 | 기술 |
|------|------|
| Data Collection | AWS Lambda, EventBridge |
| Data Storage | S3 (Data Lake), PostgreSQL (RDS) |
| Data Processing | Glue, Spark |
| ML/Analytics | SageMaker, Athena |
| Orchestration | Step Functions / Airflow |
| API Backend | FastAPI (ECS/Lambda) |
| Frontend | React / Next.js |
| Monitoring | CloudWatch, Grafana |
| IaC | Terraform |

---

## 2. 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud Infrastructure                             │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Ingestion Layer                              │   │
│  │                                                                           │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │   │
│  │   │ EventBridge │──────│   Lambda    │──────│    S3       │              │   │
│  │   │  (2분 간격) │      │ (Collector) │      │  (Raw Zone) │              │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘              │   │
│  │                              │                     │                      │   │
│  │                              ▼                     │                      │   │
│  │                        ┌─────────────┐             │                      │   │
│  │                        │ CloudWatch  │             │                      │   │
│  │                        │ (Logging)   │             │                      │   │
│  │                        └─────────────┘             │                      │   │
│  └────────────────────────────────────────────────────┼──────────────────────┘   │
│                                                       │                          │
│  ┌────────────────────────────────────────────────────┼──────────────────────┐   │
│  │                         Data Processing Layer      │                      │   │
│  │                                                    ▼                      │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │   │
│  │   │    Glue     │◀─────│Step Functions│──────│    S3       │              │   │
│  │   │ (ETL Jobs)  │      │(Orchestrator)│      │(Processed)  │              │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘              │   │
│  │         │                                         │                       │   │
│  │         ▼                                         ▼                       │   │
│  │   ┌─────────────┐                          ┌─────────────┐               │   │
│  │   │ Glue Data   │                          │   Athena    │               │   │
│  │   │  Catalog    │                          │  (Query)    │               │   │
│  │   └─────────────┘                          └─────────────┘               │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                           ML/Analytics Layer                              │   │
│  │                                                                           │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │   │
│  │   │ SageMaker   │──────│   Model     │──────│  Lambda     │              │   │
│  │   │ (Training)  │      │ (S3 Store)  │      │ (Inference) │              │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘              │   │
│  │                                                    │                      │   │
│  └────────────────────────────────────────────────────┼──────────────────────┘   │
│                                                       │                          │
│  ┌────────────────────────────────────────────────────┼──────────────────────┐   │
│  │                          Service Layer             │                      │   │
│  │                                                    ▼                      │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │   │
│  │   │   ECS       │◀─────│ API Gateway │◀─────│ CloudFront  │◀── Users    │   │
│  │   │ (FastAPI)   │      │             │      │   (CDN)     │              │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘              │   │
│  │         │                                                                 │   │
│  │         ▼                                                                 │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │   │
│  │   │   RDS       │      │ ElastiCache │      │    SNS      │              │   │
│  │   │(PostgreSQL) │      │  (Redis)    │      │ (Alerts)    │              │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘              │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   Grafana   │
                              │ (Dashboard) │
                              └─────────────┘
```

---

## 3. 데이터 파이프라인 상세

### 3.1 데이터 흐름

```
[서울시 API] ──▶ [Lambda] ──▶ [S3 Raw] ──▶ [Glue ETL] ──▶ [S3 Processed] ──▶ [Athena/ML]
     │              │             │             │                │
     │              ▼             │             ▼                ▼
     │         CloudWatch         │        Data Catalog      SageMaker
     │         (모니터링)         │        (메타데이터)      (모델 학습)
     │                            │
     ▼                            ▼
[2분 간격 수집]            [일별 파티션 저장]
                           s3://bucket/raw/year=YYYY/month=MM/day=DD/
```

### 3.2 S3 버킷 구조

```
s3://subway-delay-prediction/
├── raw/                          # 원시 데이터
│   └── year=2024/
│       └── month=05/
│           └── day=01/
│               └── train_arrival_info_10:30:00.json
│
├── processed/                    # 전처리된 데이터
│   └── subway_duration/
│       └── year=2024/
│           └── month=05/
│               └── data.parquet
│
├── features/                     # ML 피처 저장소
│   └── v1/
│       └── feature_store.parquet
│
├── models/                       # 학습된 모델
│   └── delay_prediction/
│       └── v1/
│           └── model.tar.gz
│
└── logs/                         # 로그 데이터
    └── pipeline/
```

### 3.3 데이터 스키마

#### Raw Data (JSON)
```json
{
  "collection_time": "2024-05-01T10:30:00Z",
  "data": [
    {
      "subwayId": 1002,
      "statnNm": "강남",
      "updnLine": "상행",
      "barvlDt": 120,
      "arvlMsg2": "잠실 도착",
      "arvlMsg3": "잠실",
      "recptnDt": "2024-05-01 10:29:45"
    }
  ]
}
```

#### Processed Data (Parquet)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| station | STRING | 역명 |
| prev_station | STRING | 이전역 |
| line | INT | 호선 |
| direction | STRING | 상행/하행 |
| datetime | TIMESTAMP | 수집 시간 |
| hour | INT | 시간대 (0-23) |
| duration | FLOAT | 소요시간 (초) |
| avg_duration | FLOAT | 평균 소요시간 |
| delay | FLOAT | 지연 시간 |
| is_weekend | BOOLEAN | 주말 여부 |
| is_rush_hour | BOOLEAN | 출퇴근 시간 |

---

## 4. ML 파이프라인

### 4.1 모델 아키텍처

```
[Feature Store] ──▶ [Training Pipeline] ──▶ [Model Registry] ──▶ [Inference]
      │                    │                      │                   │
      │                    ▼                      ▼                   ▼
      │              SageMaker             S3 + Versioning      Lambda/ECS
      │              Training                                   (실시간 예측)
      │
      ▼
  [Features]
  - 시간 피처 (hour, dayofweek, is_holiday)
  - 지연 피처 (lag_1~3, rolling_mean)
  - 혼잡도 피처 (승하차 인원)
  - 날씨 피처 (optional)
```

### 4.2 모델 선택

| 모델 | 용도 | 장점 |
|------|------|------|
| LightGBM | 지연 시간 예측 | 빠른 학습, 해석 가능 |
| LSTM | 시계열 패턴 | 장기 의존성 학습 |
| Prophet | 계절성 분석 | 트렌드/계절성 분리 |

### 4.3 재학습 주기

- **일일 배치**: 전일 데이터로 모델 갱신
- **주간 전체 재학습**: 전체 데이터로 모델 재학습
- **A/B 테스트**: 새 모델 vs 기존 모델 성능 비교

---

## 5. API 설계

### 5.1 엔드포인트

```
GET  /api/v1/stations                    # 역 목록
GET  /api/v1/stations/{id}/status        # 역 실시간 상태
GET  /api/v1/routes                      # 경로 검색
POST /api/v1/predict/delay               # 지연 예측
GET  /api/v1/dashboard/summary           # 대시보드 요약
WS   /api/v1/ws/realtime                 # 실시간 웹소켓
```

### 5.2 응답 예시

```json
// POST /api/v1/predict/delay
{
  "request": {
    "from_station": "강남",
    "to_station": "역삼",
    "departure_time": "2024-05-01T08:30:00"
  },
  "prediction": {
    "expected_duration": 95,
    "predicted_delay": 15,
    "confidence": 0.85,
    "delay_probability": 0.72
  }
}
```

---

## 6. 인프라 비용 추정 (월간)

### 6.1 최소 구성 (개발/테스트)
| 서비스 | 스펙 | 예상 비용 |
|--------|------|----------|
| Lambda | 2분당 1회, 512MB | ~$5 |
| S3 | 10GB | ~$0.5 |
| Glue | 1 DPU, 10분/일 | ~$5 |
| RDS | t3.micro | ~$15 |
| **합계** | | **~$25/월** |

### 6.2 운영 구성
| 서비스 | 스펙 | 예상 비용 |
|--------|------|----------|
| Lambda | 고빈도 | ~$20 |
| S3 | 100GB | ~$5 |
| Glue | 2 DPU | ~$20 |
| RDS | t3.small | ~$30 |
| ECS | 1 Task | ~$30 |
| CloudFront | 100GB | ~$10 |
| **합계** | | **~$115/월** |

---

## 7. 구현 로드맵

### Phase 1: 데이터 파이프라인 (1-2주)
- [ ] Lambda 수집기 구현
- [ ] S3 버킷 구조 설정
- [ ] EventBridge 스케줄링
- [ ] CloudWatch 모니터링

### Phase 2: 데이터 처리 (1주)
- [ ] Glue ETL Job 개발
- [ ] Data Catalog 설정
- [ ] Athena 테이블 생성

### Phase 3: ML 파이프라인 (2주)
- [ ] 피처 엔지니어링
- [ ] 모델 학습 (SageMaker)
- [ ] 모델 배포

### Phase 4: API/서비스 (2주)
- [ ] FastAPI 백엔드 개발
- [ ] API Gateway 설정
- [ ] 프론트엔드 개발

### Phase 5: 운영 (1주)
- [ ] CI/CD 파이프라인
- [ ] 모니터링 대시보드
- [ ] 알림 설정

---

## 8. 기술적 고려사항

### 8.1 확장성
- Lambda 동시 실행 제한 고려 (Reserved Concurrency)
- S3 파티셔닝으로 쿼리 성능 최적화
- 캐싱 레이어 (ElastiCache) 도입

### 8.2 장애 대응
- DLQ (Dead Letter Queue)로 실패 처리
- 재시도 로직 구현
- Multi-AZ 배포

### 8.3 보안
- IAM 역할 최소 권한 원칙
- 데이터 암호화 (S3 SSE, RDS 암호화)
- API 인증 (API Key, JWT)

---

## 9. 참고 자료

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [서울 열린데이터 광장 - 지하철 API](https://data.seoul.go.kr/)
- [AWS 서버리스 아키텍처 패턴](https://aws.amazon.com/serverless/)
