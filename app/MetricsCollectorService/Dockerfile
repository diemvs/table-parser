FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 9100

CMD ["uvicorn", "MetricsCollector:app", "--host", "0.0.0.0", "--port", "9100"]
