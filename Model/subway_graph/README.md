# SUBPICK Subway Graph

서울 지하철 환승 연결 실패 예측을 위한 그래프 기반 경로 탐색 시스템

## 개요

지하철 이용 시 **환승 연결 실패**(지연으로 인해 환승 열차를 놓치는 상황)를 예측하고,
이를 반영한 최적 경로를 추천하는 시스템입니다.

### 핵심 기능

- **환승 시간 반영**: 환승역을 호선별 별도 노드로 분리 (예: 교대_2호선 ↔ 교대_3호선)
- **실제 배차간격**: ODsay API를 통한 시간표 기반 배차간격 조회
- **환승 실패 확률**: 누적 지연 → 환승 실패 위험도 계산
- **최적 경로 추천**: 지연 비용을 반영한 Dijkstra 알고리즘

## 파일 구조

```
subway_graph/
├── graph_builder.py        # 기본 지하철 그래프 구축 (V1)
├── graph_builder_v2.py     # 환승 시간 반영 그래프 (V2)
├── delay_aware_router.py   # 지연 인식 경로 탐색기
├── odsay_transit.py        # ODsay API 클라이언트
├── subpick_router_v2.py    # 통합 라우터 (메인)
├── data/
│   ├── subway_graph_v2.json    # 그래프 데이터
│   ├── station_lines.json      # 역-호선 매핑
│   ├── transfer_stations.json  # 환승역 목록
│   └── headway_intervals.json  # 기본 배차간격
└── README.md
```

## 사용법

### 1. 환경 설정

```bash
# ODsay API 키 설정 (https://lab.odsay.com/ 에서 발급)
export ODSAY_API_KEY=your_api_key
```

### 2. 경로 추천

```python
from subpick_router_v2 import SubpickRouterV2

router = SubpickRouterV2()

# 경로 추천
result = router.recommend_route(
    start="서울역",
    end="잠실",
    departure_time="2024-01-15 18:30"
)

# 결과
# - 총 소요시간 (지연 포함)
# - 환승 실패 위험도 (%)
# - 실제 배차간격
# - 대안 경로
```

### 3. 데모 실행

```bash
python subpick_router_v2.py
```

## 알고리즘

### 환승 연결 실패 비용 계산

```
실패확률 = min(1, (누적지연 - 여유시간) / 배차간격)
추가대기 = 실패확률 × 배차간격
```

- **여유시간**: 60초 (환승 마진)
- **배차간격**: ODsay API에서 실시간 조회

### 예시

| 누적 지연 | 배차간격 | 실패확률 | 추가대기 |
|-----------|----------|----------|----------|
| 60초 | 3분 | 0% | 0초 |
| 120초 | 3분 | 33% | 60초 |
| 180초 | 3분 | 67% | 120초 |

## 시간대별 배차간격 (2호선 강남역 기준)

| 시간대 | 배차간격 |
|--------|----------|
| 08시 (출근) | 2분 40초 |
| 12시 (평시) | 5분 24초 |
| 18시 (퇴근) | 3분 41초 |
| 22시 (심야) | 6분 42초 |

## 의존성

- Python 3.8+
- requests
- pandas
- numpy

## API

### ODsay API (https://lab.odsay.com/)

- 지하철역 정보 조회
- 시간표 조회
- 경로 검색

**무료 한도**: 일 1,000건

## 향후 개선

- [ ] ML 모델 연동 (구간별 지연 예측)
- [ ] 실시간 도착정보 반영
- [ ] 웹/앱 인터페이스
