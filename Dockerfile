FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# Cloud Run 會注入 $PORT；本機測試預設 8000
ENV PORT=8000
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
