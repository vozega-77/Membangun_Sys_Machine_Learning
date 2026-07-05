from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import requests
import psutil
import threading
import pandas as pd

app = Flask(__name__)

API_URL = "http://127.0.0.1:5005/invocations"

# ==========================
# Metrik API
# ==========================
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')

LAST_PREDICTION = Gauge('last_prediction_value', 'Predicted value')
LAST_PRED_TIMESTAMP = Gauge('last_prediction_timestamp', 'Timestamp of last prediction')

# ==========================
# Metrik Sistem
# ==========================
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

# ==========================
# Background Thread for System Metrics
# ==========================
def monitor_system_metrics():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().percent)
        time.sleep(5)

threading.Thread(target=monitor_system_metrics, daemon=True).start()

# ==========================
# Inference Route
# ==========================
@app.route("/infer", methods=["POST"])
def infer():
    start = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    try:
        data = request.json
        df = pd.DataFrame(data)

        resp = requests.post(
            API_URL,
            json={"dataframe_records": df.to_dict(orient="records")}
        )

        REQUEST_LATENCY.observe(time.time() - start)

        if resp.status_code != 200:
            return jsonify({"error": resp.text}), 500

        preds = resp.json().get("predictions", [])

        # =============== FIX: Output dari MLflow adalah dict ====================
        if len(preds) > 0:
            first = preds[0]             # contoh: {"0": 0.7858}
            val = list(first.values())[0]  # ambil nilai pertama

            LAST_PREDICTION.set(float(val))
            LAST_PRED_TIMESTAMP.set(time.time())
        # =======================================================================

        return jsonify({"predictions": preds})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}


@app.route("/")
def index():
    return "Prometheus Exporter Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010)
