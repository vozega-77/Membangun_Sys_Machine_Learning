# menggunakan Random Forest (Collaborative Filtering biasa, bukan NCF)
# set Github Action & Docker Hub
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Dapatkan direktori kerja saat ini (untuk Jupyter Notebook)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Gabungkan path relatif file CSV
file_path = os.path.join(base_dir, "nilai_mahasiswa-preprocessed.csv")
print(f"✅ File CSV: {file_path}")

# --- Load dataset
df = pd.read_csv(file_path)

# --- Split data
X = df[["user", "item"]].values
y = df["rating"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Unset the tracking URI
# mlflow.set_tracking_uri("") 
# or 
# mlflow.set_tracking_uri(None) 

# ======================================================
# 🧠 MLflow Tracking
# ======================================================
# mlflow.set_experiment("CF_Mahasiswa_Sklearn") # error:  Cannot start run with ID

with mlflow.start_run(run_name="CF_RandomForest_MLProject") as run:
    print(f"🎯 MLflow Run ID: {run.info.run_id}")

    # Aktifkan autolog untuk sklearn
    mlflow.sklearn.autolog()

    # --- Model Machine Learning Collaborative Filtering
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # --- Train model
    model.fit(X_train, y_train)

    # --- Predict dan evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE: {rmse:.4f}")
    
    # log model secara eksplicit
    mlflow.sklearn.log_model(model, "model", registered_model_name="CF_Mahasiswa_Sklearn")

    # --- (Opsional) Log metrik tambahan
    mlflow.log_metric("rmse_manual", rmse)

    # Model otomatis terekam via autolog

print("🚀 Training & Logging selesai.")
