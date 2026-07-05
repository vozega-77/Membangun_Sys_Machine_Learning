# DeepLearning HyperTuning, Manual Logging (more 2 metrik than autolog)
# Up Ke Dagshub
import os
import sys
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor

import json
import matplotlib.pyplot as plt

load_dotenv()  # Token Dagshub
mlflow.set_tracking_uri("https://dagshub.com/Leo42night/smsml.mlflow")

print("Current tracking URI:", mlflow.get_tracking_uri())

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

print(f"✅ DagsHub Username: {username}")
print(f"✅ DagsHub Password/Token: {password}")

try:
    exp = mlflow.get_experiment_by_name("NCF_ManualLogging")
    if exp is None:
        exp_id = mlflow.create_experiment("NCF_ManualLogging")
        print("✅ Created experiment with ID:", exp_id)
    else:
        print("✅ Found existing experiment:", exp.name)
    print("🟢 Connection and credentials OK — write access confirmed.")
except Exception as e:
    print("❌ Connection or credentials failed:", e)
    sys.exit()

# === PATH FILE DATA ===
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

file_path = os.path.join(base_dir, "nilai_mahasiswa-preprocessed.csv")
print(f"✅ File CSV: {file_path}")
df = pd.read_csv(file_path)

# === SPLIT DATA ===
X = df[["user", "item"]].values
y = df["rating"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class SliceLayer(layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index]


# === MODEL BUILDER ===
def build_ncf_model(n_users, n_items, embed_dim=16, hidden=[32, 16, 8], lr=0.001):
    inputs = keras.Input(shape=(2,), name="user_item_input")

    user_input = SliceLayer(0)(inputs)
    item_input = SliceLayer(1)(inputs)
    user_input = layers.Reshape((1,))(user_input)
    item_input = layers.Reshape((1,))(item_input)

    user_emb = layers.Embedding(n_users, embed_dim)(user_input)
    item_emb = layers.Embedding(n_items, embed_dim)(item_input)

    user_vec = layers.Flatten()(user_emb)
    item_vec = layers.Flatten()(item_emb)
    x = layers.Concatenate()([user_vec, item_vec])

    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)

    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


# === HITUNG USER & ITEM ===
n_users = df["user"].nunique()
n_items = df["item"].nunique()

# === WRAPPER ===
regressor = KerasRegressor(
    model=lambda embed_dim, hidden, lr: build_ncf_model(
        n_users=n_users, n_items=n_items, embed_dim=embed_dim, hidden=hidden, lr=lr
    ),
    epochs=10,
    batch_size=32,
    verbose=0,
)

# === HYPERPARAMETER GRID ===
param_grid = {
    "model__embed_dim": [16],
    "model__hidden": [[64, 32, 16], [32, 16, 8]],
    "model__lr": [0.001],
    "batch_size": [32],
    "epochs": [5, 8],
}

# === MLflow ===
mlflow.set_experiment("NCF_ManualLogging")

with mlflow.start_run(run_name="GridSearch_ManualLogging"):

    # === Mulai Hitung Waktu Training ===
    t0 = time.time()

    grid = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1,
    )

    grid_result = grid.fit(X_train, y_train)

    # === Waktu training selesai ===
    train_time_seconds = time.time() - t0

    # === MODEL TERBAIK ===
    best_params = grid_result.best_params_
    best_score = grid_result.best_score_
    best_model = grid_result.best_estimator_

    # === EVALUASI ===
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    eval_result = best_model.model_.evaluate(X_test, y_test, verbose=0)

    if isinstance(eval_result, (list, tuple)):
        test_loss = eval_result[0]
        test_mae = eval_result[1] if len(eval_result) > 1 else np.nan
        test_mse = eval_result[2] if len(eval_result) > 2 else np.nan
    else:
        test_loss, test_mae, test_mse = eval_result, np.nan, np.nan

    # === LOG PARAMETER ===
    mlflow.log_params(
        {
            "embed_dim": best_params["model__embed_dim"],
            "hidden_layers": best_params["model__hidden"],
            "learning_rate": best_params["model__lr"],
            "batch_size": best_params["batch_size"],
            "epochs": best_params["epochs"],
            "n_users": n_users,
            "n_items": n_items,
            "optimizer": "Adam",
            "loss": "mse",
        }
    )

    # === PARAMETER TAMBAHAN (Manual & Tidak Ada di Autolog) ===
    total_params = best_model.model_.count_params()
    mlflow.log_metric("model_total_params", total_params)
    mlflow.log_metric("train_time_seconds", train_time_seconds)

    # === TRAINING METRICS ===
    history = getattr(best_model.model_, "history", None)
    if history and hasattr(history, "history"):
        hist_dict = history.history
        for metric_name, values in hist_dict.items():
            mlflow.log_metric(f"train_final_{metric_name}", float(values[-1]))
    else:
        mlflow.log_metric("train_final_loss", np.nan)

    # === 2 METRIK TAMBAHAN WAJIB (Manual) ===
    mlflow.log_metric("best_cv_neg_mse", best_score)
    mlflow.log_metric("test_rmse", rmse)

    # === METRIK TEST ===
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", test_mse)
    
    mlflow.keras.log_model(best_model.model_, "model")

    # === SIMPAN MODEL KE ARTIFACT ===
    best_model.model_.save("ncf_model_best.h5")
    mlflow.log_artifact("ncf_model_best.h5", artifact_path="model")
    
    # === Model Summary ===
    with open("model_summary.txt", "w") as f:
        best_model.model_.summary(print_fn=lambda x: f.write(x + "\n"))
    mlflow.log_artifact("model_summary.txt", artifact_path="model_info")
    
    # === Model Architecture JSON ===
    with open("architecture.json", "w") as f:
        f.write(best_model.model_.to_json())
    mlflow.log_artifact("architecture.json", artifact_path="model_info")

    # === SAFE HISTORY EXTRACTION ===
    history_obj = getattr(best_model.model_, "history", None)
    history = history_obj.history if history_obj and hasattr(history_obj, "history") else {}

    # === SAVE TRAINING HISTORY JSON ===
    with open("training_metrics.json", "w") as f:
        json.dump(history, f, indent=4)
    mlflow.log_artifact("training_metrics.json", artifact_path="training_info")
    
    metrics_dict = {
        "best_cv_neg_mse": best_score,
        "test_rmse": rmse,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "model_total_params": total_params,
        "train_time_seconds": train_time_seconds,
    }

    # Simpan ke file
    with open("metrics_manual.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Upload ke MLflow sebagai artifact
    mlflow.log_artifact("metrics_manual.json")

    # === TRAINING LOSS PLOT ===
    if "loss" in history:
        plt.plot(history["loss"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("training_loss.png")
        mlflow.log_artifact("training_loss.png", artifact_path="metrics")
    else:
        print("⚠️ History training kosong — mungkin GridSearch tidak menyimpan history.")

    # === OUTPUT TERMINAL ===
    print("\n🏆 Best Params:", best_params)
    print(f"📉 Best Score (CV neg MSE): {best_score:.4f}")
    print(f"📉 Test RMSE: {rmse:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f} — MAE: {test_mae:.4f} — MSE: {test_mse:.4f}")
    print(f"⏱️ Train Time: {train_time_seconds:.2f} sec")
    print(f"📦 Total Params: {total_params}")

print("🚀 Training, manual logging, dan push ke DagsHub selesai.")
