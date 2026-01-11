#!/bin/bash
# AWS SAM 배포 스크립트

set -e

# 환경 설정
ENVIRONMENT=${1:-dev}
STACK_NAME="subway-delay-prediction-${ENVIRONMENT}"
REGION="ap-northeast-2"

echo "==================================="
echo "Deploying to ${ENVIRONMENT} environment"
echo "Stack: ${STACK_NAME}"
echo "Region: ${REGION}"
echo "==================================="

# API 키 확인
if [ -z "$SEOUL_API_KEY" ]; then
    echo "Error: SEOUL_API_KEY environment variable is not set"
    echo "Usage: SEOUL_API_KEY=your_key ./deploy.sh [dev|prod]"
    exit 1
fi

# SAM 빌드
echo "Building Lambda function..."
sam build --template-file template.yaml

# SAM 배포
echo "Deploying stack..."
sam deploy \
    --stack-name $STACK_NAME \
    --region $REGION \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        SeoulApiKey=$SEOUL_API_KEY \
        Environment=$ENVIRONMENT \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

echo "==================================="
echo "Deployment complete!"
echo "==================================="

# 출력값 표시
echo "Stack outputs:"
aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs' \
    --output table
