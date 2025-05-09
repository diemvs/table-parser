from fastapi import FastAPI, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil
import time

# Инициализация
app = FastAPI()

# CPU метрики
cpu_usage_gauge = Gauge('cpu_usage_percent', 'CPU Usage Percent')
memory_usage_gauge = Gauge('memory_usage_percent', 'Memory Usage Percent')

# Обновление метрик
@app.on_event("startup")
async def start_metrics_collector():
    from threading import Thread

    def collect_metrics():
        while True:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1.0)
            cpu_usage_gauge.set(cpu_percent)
            memory = psutil.virtual_memory()
            memory_usage_gauge.set(memory.percent)
            time.sleep(5)

    Thread(target=collect_metrics, daemon=True).start()

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
