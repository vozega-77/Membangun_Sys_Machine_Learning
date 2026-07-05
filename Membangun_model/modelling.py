import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Dapatkan direktori file CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "nilai_mahasiswa-preprocessed.csv")

# Load dataset
df = pd.read_csv(file_path)

X = df[["user", "item"]].values
y = df["rating"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment("CF_Mahasiswa_Sklearn")

with mlflow.start_run(run_name="ML_CF_RandomForest_autolog") as run:
    print(f"🎯 MLflow Run ID: {run.info.run_id}")

    # Auto log (model akan otomatis tersimpan dalam folder 'model')
    mlflow.sklearn.autolog(log_models=True)

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Eval
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse_manual", rmse)

print("🚀 Training & Logging autolog selesai.")
