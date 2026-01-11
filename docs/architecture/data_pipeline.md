# 데이터 파이프라인 상세 설계

## 1. 파이프라인 개요

### 1.1 파이프라인 유형
- **Batch Pipeline**: 역간 소요시간 집계 (일일)
- **Near Real-time Pipeline**: 실시간 열차 위치 수집 (2분 간격)

### 1.2 데이터 소스
| 소스 | API | 수집 주기 | 용도 |
|------|-----|----------|------|
| 서울시 실시간 도착정보 | realtimeStationArrival | 2분 | 열차 위치, 도착 예정 |
| 서울시 혼잡도 정보 | realtimePosition | 5분 | 칸별 혼잡도 |
| 기상청 날씨 | (optional) | 1시간 | 날씨 영향 분석 |

---

## 2. Lambda 수집기 설계

### 2.1 아키텍처

```
EventBridge (2분 간격)
        │
        ▼
  ┌─────────────┐
  │   Lambda    │──────▶ S3 Raw Zone
  │ (collector) │
  └─────────────┘
        │
        ├──▶ CloudWatch Logs
        └──▶ CloudWatch Metrics
```

### 2.2 Lambda 코드 구조

```python
# lambda/collector/handler.py

import json
import boto3
import requests
from datetime import datetime
import os

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    서울시 지하철 실시간 도착정보 수집 Lambda
    """
    # 환경 변수
    API_KEY = os.environ['SEOUL_API_KEY']
    BUCKET_NAME = os.environ['S3_BUCKET']

    # API 호출
    url = f"http://swopenAPI.seoul.go.kr/api/subway/{API_KEY}/json/realtimeStationArrival/ALL"
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code}")

    data = response.json()

    # S3 저장 경로 생성
    now = datetime.now()
    s3_key = (
        f"raw/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"train_arrival_{now.strftime('%H:%M:%S')}.json"
    )

    # 메타데이터 추가
    payload = {
        "collection_time": now.isoformat(),
        "source": "seoul_metro_api",
        "record_count": len(data.get('realtimeArrivalList', [])),
        "data": data.get('realtimeArrivalList', [])
    }

    # S3 업로드
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(payload, ensure_ascii=False),
        ContentType='application/json'
    )

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Data collected successfully',
            's3_key': s3_key,
            'record_count': payload['record_count']
        })
    }
```

### 2.3 Lambda 설정

```yaml
# serverless.yml 또는 SAM template

AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  SubwayCollectorFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handler.lambda_handler
      Runtime: python3.11
      Timeout: 60
      MemorySize: 256
      Environment:
        Variables:
          SEOUL_API_KEY: !Ref SeoulApiKey
          S3_BUCKET: !Ref DataBucket
      Policies:
        - S3WritePolicy:
            BucketName: !Ref DataBucket
      Events:
        ScheduleEvent:
          Type: Schedule
          Properties:
            Schedule: rate(2 minutes)
            Enabled: true

  DataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: subway-delay-prediction-data
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: MoveToGlacier
            Status: Enabled
            Transitions:
              - TransitionInDays: 90
                StorageClass: GLACIER
```

---

## 3. Glue ETL 설계

### 3.1 ETL 흐름

```
S3 Raw Zone ──▶ Glue Job ──▶ S3 Processed Zone
                   │
                   ├── 데이터 정제
                   ├── 스키마 변환
                   ├── 집계 계산
                   └── 피처 생성
```

### 3.2 Glue Job 코드

```python
# glue/etl_daily_aggregation.py

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_INPUT_PATH', 'S3_OUTPUT_PATH'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# 1. Raw 데이터 읽기
raw_df = spark.read.json(args['S3_INPUT_PATH'])

# 2. 데이터 정제
cleaned_df = raw_df.select(
    F.col("collection_time").cast("timestamp"),
    F.explode("data").alias("record")
).select(
    F.col("collection_time"),
    F.col("record.statnNm").alias("station"),
    F.col("record.arvlMsg3").alias("prev_station"),
    F.col("record.subwayId").alias("line_id"),
    F.col("record.updnLine").alias("direction"),
    F.col("record.barvlDt").cast("int").alias("arrival_time"),
    F.col("record.arvlCd").cast("int").alias("arrival_code"),
    F.col("record.recptnDt").cast("timestamp").alias("record_time")
)

# 3. 시간 피처 추가
featured_df = cleaned_df.withColumn(
    "hour", F.hour("collection_time")
).withColumn(
    "date", F.to_date("collection_time")
).withColumn(
    "dayofweek", F.dayofweek("collection_time")
).withColumn(
    "is_weekend", F.when(F.col("dayofweek").isin([1, 7]), 1).otherwise(0)
)

# 4. 역간 소요시간 집계 (시간대별 평균)
aggregated_df = featured_df.groupBy(
    "date", "station", "prev_station", "line_id", "direction", "hour"
).agg(
    F.avg("arrival_time").alias("avg_duration"),
    F.stddev("arrival_time").alias("std_duration"),
    F.count("*").alias("sample_count"),
    F.min("arrival_time").alias("min_duration"),
    F.max("arrival_time").alias("max_duration")
)

# 5. 지연 피처 계산
window_spec = Window.partitionBy("station", "prev_station", "hour", "dayofweek")
final_df = aggregated_df.withColumn(
    "overall_avg", F.avg("avg_duration").over(window_spec)
).withColumn(
    "delay", F.col("avg_duration") - F.col("overall_avg")
)

# 6. Parquet 저장 (파티션)
final_df.write.mode("overwrite").partitionBy("date").parquet(args['S3_OUTPUT_PATH'])

job.commit()
```

### 3.3 Glue 스케줄링

```yaml
# Step Functions 또는 Glue Workflow로 스케줄링

GlueWorkflow:
  Name: subway-daily-etl
  Triggers:
    - Name: daily-trigger
      Type: SCHEDULED
      Schedule: cron(0 6 * * ? *)  # 매일 06:00 UTC (15:00 KST)
      Actions:
        - JobName: subway-etl-job
```

---

## 4. 데이터 품질 체크

### 4.1 품질 규칙

```python
# glue/data_quality_checks.py

from awsglue.context import GlueContext
from awsgluedq.transforms import EvaluateDataQuality

# 품질 규칙 정의
rules = """
Rules = [
    RowCount > 1000,
    IsComplete "station",
    IsComplete "prev_station",
    ColumnValues "avg_duration" between 30 and 300,
    ColumnValues "hour" between 5 and 23,
    Uniqueness "station, prev_station, date, hour" > 0.95
]
"""

# 품질 평가 실행
quality_result = EvaluateDataQuality().process_rows(
    frame=dynamic_frame,
    ruleset=rules,
    publishing_options={
        "dataQualityEvaluationContext": "subway_data_quality",
        "enableDataQualityCloudWatchMetrics": "true",
        "enableDataQualityResultsPublishing": "true"
    }
)
```

### 4.2 알림 설정

```python
# 품질 실패 시 SNS 알림
def send_quality_alert(failed_rules):
    sns = boto3.client('sns')
    sns.publish(
        TopicArn='arn:aws:sns:ap-northeast-2:xxx:data-quality-alerts',
        Message=json.dumps({
            'alert_type': 'DATA_QUALITY_FAILURE',
            'failed_rules': failed_rules,
            'timestamp': datetime.now().isoformat()
        }),
        Subject='[Alert] 지하철 데이터 품질 이슈 발생'
    )
```

---

## 5. 모니터링 대시보드

### 5.1 CloudWatch 메트릭

```python
# Lambda 내 커스텀 메트릭 발행
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='SubwayDataPipeline',
    MetricData=[
        {
            'MetricName': 'RecordsCollected',
            'Value': record_count,
            'Unit': 'Count',
            'Dimensions': [
                {'Name': 'Pipeline', 'Value': 'collector'}
            ]
        },
        {
            'MetricName': 'CollectionLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds',
            'Dimensions': [
                {'Name': 'Pipeline', 'Value': 'collector'}
            ]
        }
    ]
)
```

### 5.2 알람 설정

```yaml
# CloudWatch Alarms
Alarms:
  - Name: CollectorFailure
    Metric: Errors
    Threshold: 3
    Period: 300
    EvaluationPeriods: 2
    Action: SNS Topic

  - Name: LowRecordCount
    Metric: RecordsCollected
    Threshold: 500
    ComparisonOperator: LessThanThreshold
    Period: 300
    Action: SNS Topic
```

---

## 6. 데이터 카탈로그 (Glue Data Catalog)

### 6.1 테이블 정의

```sql
-- Athena에서 조회 가능한 테이블

CREATE EXTERNAL TABLE subway_raw (
    collection_time TIMESTAMP,
    station STRING,
    prev_station STRING,
    line_id INT,
    direction STRING,
    arrival_time INT,
    arrival_code INT
)
PARTITIONED BY (year INT, month INT, day INT)
STORED AS PARQUET
LOCATION 's3://subway-delay-prediction/raw/';

CREATE EXTERNAL TABLE subway_processed (
    date DATE,
    station STRING,
    prev_station STRING,
    line_id INT,
    direction STRING,
    hour INT,
    avg_duration DOUBLE,
    std_duration DOUBLE,
    sample_count INT,
    delay DOUBLE
)
PARTITIONED BY (date DATE)
STORED AS PARQUET
LOCATION 's3://subway-delay-prediction/processed/';
```

### 6.2 샘플 쿼리

```sql
-- 특정 역의 시간대별 평균 지연
SELECT
    hour,
    AVG(delay) as avg_delay,
    COUNT(*) as data_points
FROM subway_processed
WHERE station = '강남' AND prev_station = '역삼'
GROUP BY hour
ORDER BY hour;

-- 가장 지연이 심한 구간 Top 10
SELECT
    station,
    prev_station,
    AVG(delay) as avg_delay
FROM subway_processed
WHERE date >= date_add('day', -7, current_date)
GROUP BY station, prev_station
ORDER BY avg_delay DESC
LIMIT 10;
```

---

## 7. 배포 자동화 (CI/CD)

### 7.1 GitHub Actions

```yaml
# .github/workflows/deploy-pipeline.yml

name: Deploy Data Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'lambda/**'
      - 'glue/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-northeast-2

      - name: Deploy Lambda
        run: |
          cd lambda/collector
          pip install -r requirements.txt -t .
          zip -r function.zip .
          aws lambda update-function-code \
            --function-name subway-collector \
            --zip-file fileb://function.zip

      - name: Deploy Glue Job
        run: |
          aws s3 cp glue/etl_daily_aggregation.py \
            s3://subway-delay-prediction/scripts/
          aws glue update-job \
            --job-name subway-etl-job \
            --job-update "Command={ScriptLocation=s3://subway-delay-prediction/scripts/etl_daily_aggregation.py}"
```

---

## 8. 비용 최적화

### 8.1 Lambda
- Reserved Concurrency로 동시 실행 제한
- ARM64 아키텍처 사용 (20% 비용 절감)

### 8.2 S3
- Intelligent-Tiering 활성화
- 90일 이상 데이터 Glacier로 이동

### 8.3 Glue
- 필요한 DPU만 사용 (auto-scaling)
- Spark UI 비활성화로 비용 절감

---

## 9. 다음 단계

1. **Terraform으로 IaC 구현**
2. **Airflow 도입 (복잡한 의존성 관리)**
3. **Data Lineage 추적 (OpenLineage)**
4. **Feature Store 구축 (SageMaker Feature Store)**
