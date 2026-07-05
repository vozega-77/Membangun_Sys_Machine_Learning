from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd

app = Flask(__name__)

# Endpoint MLflow Model Serve
API_URL = "http://127.0.0.1:5005/invocations"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)

        # request ke MLflow model serve
        response = requests.post(
            API_URL,
            json={"dataframe_records": df.to_dict(orient="records")}
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), 500

        preds = response.json().get("predictions", [])

        return jsonify({"predictions": preds})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return "Inference endpoint OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8005)