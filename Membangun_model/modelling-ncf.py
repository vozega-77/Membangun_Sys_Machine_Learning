# Deep Learning CF
# model dasar NCF untuk modelling_tuning.py
# hanya pakai autolog
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor

from tensorflow.keras.saving import register_keras_serializable

# Dapatkan direktori kerja saat ini (untuk Jupyter Notebook)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Gabungkan path relatif file CSV
file_path = os.path.join(base_dir, "nilai_mahasiswa-preprocessed.csv")
print(f"✅ File CSV: {file_path}")
# Load dataset
df = pd.read_csv(file_path)

# --- Split data
X = df[["user", "item"]].values
y = df["rating"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@register_keras_serializable()
class SliceLayer(layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

# --- Model builder
def build_ncf_model(n_users, n_items, embed_dim=16, hidden=[32, 16, 8]):
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
    model.compile(optimizer="adam", loss="mse")
    return model


# --- Setup model
n_users = df["user"].nunique()
n_items = df["item"].nunique()

model = KerasRegressor(
    model=lambda: build_ncf_model(n_users, n_items), epochs=10, batch_size=32, verbose=1
)

# ======================================================
# 🧠 MLflow Tracking
# ======================================================

# Set URI lokal (atau bisa diganti ke server remote)
# mlflow.set_tracking_uri("file://" + os.path.join(base_dir, "mlruns"))
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set nama eksperimen
mlflow.set_experiment("NCF_ManualLogging")

with mlflow.start_run(run_name="build_modelling") as run:
    print(f"🎯 MLflow Run ID: {run.info.run_id}")
    # Aktifkan autolog untuk TensorFlow
    mlflow.tensorflow.autolog()

    # --- Train model
    model.fit(X_train, y_train)

    # --- Predict dan evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE: {rmse:.4f}")

    # --- Log metrik manual juga (opsional)
    mlflow.log_metric("rmse", rmse)
    
    # izinkan unsafe deserialization ketika nanti model di load.
    keras.config.enable_unsafe_deserialization()

    # --- Simpan model
    mlflow.tensorflow.log_model(
        model.model_,  # scikeras wrapper punya atribut .model_
        name="model",
        registered_model_name="NCF_ManualLogging",
    )

print("🚀 Training & Logging selesai.")
