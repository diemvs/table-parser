FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9200

CMD ["uvicorn", "PreprocessService:app", "--host", "0.0.0.0", "--port", "9200"]
