FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# libgomp is needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modelledger ./modelledger
COPY config ./config
COPY scripts ./scripts

RUN mkdir -p data anchors models reports

EXPOSE 8000 8001
CMD ["uvicorn", "modelledger.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
