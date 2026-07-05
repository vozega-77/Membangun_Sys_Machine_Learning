import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# Dapatkan direktori kerja saat ini (untuk Jupyter Notebook)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Gabungkan path relatif file CSV
file_path = os.path.join(base_dir, "./../nilai_mahasiswa_raw.csv")
print(f"✅ File CSV: {file_path}")
# Load dataset
df = pd.read_csv(file_path)

## EDA
print("\n--- EDA ---")
print(f"{df['nim'].nunique()} nim, {df['matkul'].nunique()} maktul")
print(df[["matkul", "nim", "nilai_angka"]].isna().sum())
print(f"nilai_angka: {df['nilai_angka'].unique().tolist()}")
duplicate_nim_matkul = df[df.duplicated(subset=["nim", "matkul"], keep=False)]
print(
    "Jumlah pasangan 'nim' dan 'matkul' yang memiliki duplikasi:",
    duplicate_nim_matkul.shape[0],
)

#### Preprocessing
# --- CLEANING ---
# Hapus kolom yang tidak diperlukan
df = df.drop(columns=["dosen", "no", "mahasiswa", "nilai", "nilai_huruf", "prodi"])

# Convert 'nilai_angka' to numeric, coercing errors to NaN
df["nilai_angka"] = pd.to_numeric(df["nilai_angka"], errors="coerce")

# Sort by 'nilai_angka' (NaNs will be at the end) and then drop duplicates
# This keeps the row with a valid numeric value if it exists, otherwise keeps the first row
df = df.sort_values(by=["nim", "matkul", "nilai_angka"], na_position="last")
df.drop_duplicates(subset=["nim", "matkul"], keep="first", inplace=True)

# Drop rows where 'nilai_angka' is still NaN after duplicate removal
df.dropna(subset=["nilai_angka"], inplace=True)

# Convert 'nilai_angka' to float
df["nilai_angka"] = df["nilai_angka"].astype(float)

# -- after preprocessing check --
print("\n--- after preprocessing check ---")
print(f"{df['nim'].nunique()} nim, {df['matkul'].nunique()} maktul")
print(df[["matkul", "nim", "nilai_angka"]].isna().sum())
print(f"nilai_angka: {df['nilai_angka'].unique().tolist()}")
# check for duplicate 'nim' and 'matkul' pairs
duplicate_nim_matkul = df[df.duplicated(subset=["nim", "matkul"], keep=False)]
print(
    "Jumlah pasangan 'nim' dan 'matkul' yang memiliki duplikasi:",
    duplicate_nim_matkul.shape[0],
)

# Encode mahasiswa dan mata kuliah
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["user"] = user_encoder.fit_transform(df["nim"])
df["item"] = item_encoder.fit_transform(df["matkul"])

# Normalisasi nilai_angka menjadi 0-1
scaler = MinMaxScaler()
df["rating"] = scaler.fit_transform(df[["nilai_angka"]])

# Dataset siap untuk NCF
print(df[["user", "item", "rating"]].head())

# (Bukti berhasil di export ke CSV)
file_path_export = os.path.join(base_dir, "nilai_mahasiswa-preprocessed.csv")
df.to_csv(file_path_export, index=False)
print(
    "✅ Data hasil preprocessing berhasil disimpan sebagai 'nilai_mahasiswa-preprocessed.csv'"
)
