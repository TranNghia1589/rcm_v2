FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/api.txt -r requirements/ml.txt

COPY . .

EXPOSE 8081

CMD ["python", "-m", "uvicorn", "services.embedding.app.main:app", "--host", "0.0.0.0", "--port", "8081"]
