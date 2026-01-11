# Lambda 로컬 테스트용 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY handler.py .

# 기본 명령어
CMD ["python", "handler.py", "--help"]
