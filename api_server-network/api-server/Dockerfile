# ✅ Python 3.10 slim 이미지 사용
FROM python:3.10.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 먼저 복사 → 캐시 재사용을 위해
COPY requirements.txt .

# 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN apt-get update && apt-get install -y curl

# 전체 코드 복사 (볼륨 마운트 시 실제 사용되진 않음, fallback용)
COPY . .

# FastAPI + Uvicorn 실행 (개발용 핫리로드 포함)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
