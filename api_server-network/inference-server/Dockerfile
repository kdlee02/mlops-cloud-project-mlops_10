FROM python:3.10.12-slim

WORKDIR /app

# requirements.txt 먼저 복사해서 캐싱
COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y curl
# 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt

# 빈 모델 디렉토리 생성
RUN mkdir -p /app/model

# 코드 복사는 volume으로 대체되므로 fallback용
COPY src/ ./src/

# 개발 서버 실행
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
