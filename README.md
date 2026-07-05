# Submisi Membangun Sistem Machine Learning
Modul pembelajaran dari kelas [Dicoding](https://www.dicoding.com/): **Membangun Sistem Machine Learning**<br>
Sub Repo:
- Preprocessing: https://github.com/vozega-77/Membangun_Sys_Machine_Learning/tree/main/Eksperimen_SML_Invokavitzega
- CI/CD: https://github.com/vozega-77/Membangun_Sys_Machine_Learning/tree/main/Workflow-CI/MLProject


## Kriteria
1. Basic, Skilled, Advance Terpenuhi: File automasi preprocessing (`preprocessing/`) mengembalikan data latih, run di Github Action save di repo.
2. Advance Terpenuhi: DeepLearning Model Tuning (`Membanung_model/modelling_tuning.py`) pakai MLFlow Track UI & save di DagsHub. Pakai autolog dan minimal 2 metrik tambahan.
3. Advance Terpenuhi: Automasi modelling (`MLProject/`) save ke Github repo dan DockerHub.
4. Advance Terpenuhi: 3 metrik Prometheus, 10 Metrik Grafana, 3 Alert Grafana.

## Cara Pakai
NOTE: Saya menggunakan WSL yang berbasis Linux (Window peru penyesuaian script untuk slash `/`)
### Setup
```bash
# setup lingkungan
conda create -n ncf-env python=3.12
conda activate ncf-env

pip install -r requirements.txt
# --- Didapat dari pip freeze > requirements.txt
# jika gagal pakai file requiremets.txt
conda env create -f environment.yml
# didapat dari conda env export > environment.yml 
# Jika masih gagal, manual install.
pip numpy pandas scikit-learn tensorflow scikeras mlflow uvicorn fastapi psutil
# warning: tensorflow butuh 600MB. Total size env adalah 3.0G
```

### Kriteria 1: Preprocessing
```bash
# test preprocessing (kriteria 1 Advance)
python preprocessing/automate_invosmartplay.py 
```
Ketika push ke Github (auto run Github Action, save ke Github Repo).

### Kriteria 2: Modelling
Perlu run Mlflow Server lebih dulu:
```bash
mlflow server --host 127.0.0.1 --port 5001
```
- Single modelling
```bash
# run script (mode Autolog)
# -- sklearn.randomForest
python Membangun_model/modelling.py 
# -- tensorflow
python Membangun_model/modelling-ncf.py 

```
- Hypertuning ManualLog, auto up ke dagshub. Gunakan contoh di `.env.production`, buat salinan di `.env`. Gunakan token Dagshub sebagai password.
```bash 
# run script tuning (tensorflow)
python Membangun_model/modelling_tuning.py 
```

### Kriteria 3: CI/CD
- Uji modelling di local `MLProject\modelling.py` (hapus folder `/mlruns` untuk reset config mlflow-artifacts URI, jika tidak akan dapat error Tracking URI [lihat **log.txt**]):
```bash
mlflow run MLProject --env-manager=local
# akan membuat folder dan data model mlruns/
# === Run (ID 'f5974476ca3740ae92a4e5d1fb34b62e') succeeded ===
```
- Uji docker di local:
```bash
# build image (5.6GB) (tidak perlu run mlflow server)
mlflow models build-docker \
  -m models:/CF_Mahasiswa_Sklearn/1 \
  -n <dockerhub_username>/cf_rekomendasi:latest

# run image (tidak perlu setup conda env, karena mlflow dan semua dependency load dari container)
# - berhasil jika tampil pesan: "INFO:     Application startup complete."
docker run -p 5000:8000 \
  --entrypoint mlflow \
  <dockerhub_username>/cf_rekomendasi:latest \
  models serve -m /opt/ml/model -h 0.0.0.0 -p 8000

# info: 5000 adalah host, 8000 adalah internal docker

# test inferensi (localhost komputer akses host docker)
curl -X POST http://127.0.0.1:5000/invocations \
-H "Content-Type: application/json" \
-d '{
      "dataframe_split": {
        "columns": ["user", "item"],
        "data": [[1, 208]]
      }
    }'

# (test) deploy manual to Docker Hub
docker login # koneksi device, buat DockerHub repo 'cf_rekomendasi'
docker push <dockerhub_username>/cf_rekomendasi:latest # makan waktu +10 menit (5.6GB)
```
- Sebelum push ke Repo Github, tambahkan secret di repo github:
  - **USERNAME** (username github)
  - **EMAIL** (email akun github)
  - **DOCKER_USERNAME**
  - **DOCKER_PASSWORD**

### Kriteria 4: Log & Alert
Beberapa server yang akan djalankan (**Flask Inference API** hanya opsi untuk test deployment, dapat di-skip)

| Komponen | Perintah | Fungsi | Port |
| -------- | -------- | ------ | ---- |
| **MLflow Tracking Server** | `mlflow server --host 127.0.0.1` | UI untuk eksperimen, log, dan registry model | `5001` |
| **MLflow Model Serve** | `mlflow models serve ... -p 5005` | Menyajikan model untuk prediksi | `5005` |
| *Flask Inference API* | *`python inference.py`* | REST API untuk prediksi model *(opsional)* | *`8005`* |
| **Prometheus Exporter** | `python prometheus_exporter.py` | Setup endpoint `/metrics` untuk Prometheus | `8010` |
| **Prometheus** | `prometheus.exe --config.file=prometheus.yml` | Scrape metrics dari API lain | `9090` |
| **Grafana** | (jalankan grafana) | Dashboard visualisasi | `3000` |

```
Client → inferensi.py → (HTTP request) → MLflow Model Serve
                                 ↓
                         prometheus_exporter.py
                          (mengukur & expos metrics)
```

### 1. Run Model
Sebelum run model, run dulu mlflow tracking server:
```bash
conda activate ncf-env
mlflow server --host 127.0.0.1
```

- **Mlflow Model Serve**: Akan membuat REST API yang diakses melalui endpoint `/invocations`.
```bash
# 1. anda pelu set variabel MLFLOW_TRACKING_URI
# CMD: set MLFLOW_TRACKING_URI=http://127.0.0.1:5000 # atau lewat global Variable
# CMD: cek dengan echo %MLFLOW_TRACKING_URI%

# 2. Pilih versi (lihat di `mlruns\models\<nama_model>\version-x`). Silakan pilih diantara 2 model berikut:
# - Model NCF (🌟 MODEL UTAMA)
# Pastikan model sudah dibuat mengikuti Kriteria 2 (**modelling-ncf** atau **modelling_tuning.py**)
mlflow models serve -m "models:/NCF_ManualLogging/1" -p 5005 --no-conda

# - Model CF (Opsional/Tambahan): Random Forest
# Pastikan model sudah dibuat mengikuti kriteria 2 (modelling.py) atau Kriteria 3.
# mlflow models serve -m "models:/CF_Mahasiswa_Sklearn/1" -p 5005 --no-conda
```
Test API MLFlow Model Serve:
```bash
curl -X POST http://127.0.0.1:5005/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["user", "item"],
          "data": [[1, 208]]
        }
      }'
# {"predictions": [{"0": 0.89720219373703}]}(base)
```

- Server Inferensi
Default menggunakan REST API dari server model **models:/NCF_ManualLogging/1** 
```bash 
python "Monitoring dan Logging/7.inference.py" 
# INFO: inference.py hanya untuk inferensi → tidak ada Prometheus di dalamnya.
```
Test API Server Inferensi:
```bash
curl -X POST http://127.0.0.1:8005/predict \
  -H "Content-Type: application/json" \
  -d '[
        {"user": 1, "item": 10},
        {"user": 2, "item": 20},
        {"user": 3, "item": 30}
      ]'
# {"predictions":[0.7424633502960205,0.8271910548210144,0.7336715459823608]}
```

### 2. Prometheus
[Download](https://prometheus.io/download/) & setup path. Lihat Hasil: http://localhost:9090/targets
```bash
# sebelum run ini, pasitikan model server sudah run.
python "Monitoring dan Logging/3.prometheus_exporter.py"
C:\prometheus\prometheus-3.5.0.windows-amd64\prometheus.exe --config.file="Monitoring dan Logging\2.prometheus.yml"
```

Test API Server Prometeus Exporter (1348 user, 211 item):
```bash
curl -X GET http://127.0.0.1:8010 -w "\n"
curl -X POST http://127.0.0.1:8010/infer \
  -H "Content-Type: application/json"  -w "\n" \
  -d '[
        {"user": 1, "item": 10},
        {"user": 2, "item": 20},
        {"user": 3, "item": 30}
      ]'
# {"predictions":[{"0":0.7858941555023193},{"0":0.7942813634872437},{"0":0.8206713199615479}]}
```

| Metric                          | Valid | Catatan          |
| ------------------------------- | ----- | ---------------- |
| `http_requests_total`           | ✔     | Counter          |
| `http_request_duration_seconds` | ✔     | Histogram        |
| `http_requests_throughput`      | ✔     | Gunakan `rate()` |
| `system_cpu_usage`              | ✔     | Butuh `psutil`   |
| `system_ram_usage`              | ✔     | Butuh `psutil`   |

### 3. grafana
- [Download program](https://grafana.com/grafana/download) & setup path.
- http://localhost:3000/ adalah default, jika tidak bisa, pakai port lain.
```bash
cd "C:\Program Files\GrafanaLabs\grafana\bin"
grafana-server.exe --homepath "C:\Program Files\GrafanaLabs\grafana"
# --homepath? Grafana butuh mengetahui home directory untuk menemukan file konfigurasi default.
``` 
- By default, kredensial masuk dengan username “admin” dan password “admin”.
> Grafana bisa saja otomatis berjalan di latar belakang, untuk mengeceknya (lihat tips)

**📃 Note:** Alert hanya 1, karena ketika menambahkan alert kedua, dapat error uuid kosong (lihat **Log.txt**).

## Disk Usage Managemet
```bash
# check disk usave of some (env, etc.)
du -hs /<path>

# Info: Setiap pertama run `mlflow models serve` dengan model baru akan membuat environment python (Conda) baru dari conda.yaml
# Periksa env dan remove jika sudah yakin test berhasil dan ingin hemat ruang.
conda env list # lihat list env conda
conda remove --name <nama_env> --all
```

## Tips
- Di CMD: Untuk terminate program jika **Ctrl+C** tidak bekerja, bisa pakai **Ctrl+Pause/Break**.
- Grafana berjalan di latar belakang, jika ingin restart (ada perubahan konfigurasi) atau ingin stop server nya, tinggal cari `services` pada pencarian windows.
- Cek Port terpakai: `netstat -ano | findstr "<nomor_port>"` atau `netstat -ano | find /i "<nomor_port>"`
