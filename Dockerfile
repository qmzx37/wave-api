FROM python:3.12-slim

WORKDIR /app

# (선택) 기본 유틸
RUN pip install --no-cache-dir --upgrade pip

# 의존성 먼저 설치 (캐시 효율)
COPY requirement-lite.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 앱 코드 복사
COPY . /app

# 컨테이너 내부 포트
EXPOSE 8000

# uvicorn 실행 (컨테이너 안에서는 8000 고정 권장)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]