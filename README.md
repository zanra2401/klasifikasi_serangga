# Insect Sound Classifier - Streamlit App

Aplikasi Streamlit untuk klasifikasi jenis serangga berdasarkan wingbeat menggunakan model two-stage (RandomForest + SVM).

## Setup & Instalasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pastikan Model Tersedia
Model harus berada di direktori parent (`../`):
- `random_forest.pkl` - Model RandomForest global
- `svc.pkl` - Model SVM untuk refinement Aedes/Quinx female

### 3. Jalankan Streamlit
```bash
streamlit run app.py
```

Aplikasi akan membuka di browser (default: http://localhost:8501)

## Fitur

✅ **Upload Data:**
- Format ARFF (UCR/UEA time-series)
- Format CSV (600+ kolom untuk time-series)

✅ **Two-Stage Classification:**
- Stage 1: RandomForest untuk semua 10 spesies
- Stage 2: SVM refinement untuk Aedes_female & Quinx_female

✅ **Hasil Prediksi:**
- Summary tabel dengan count & persentase
- Detail prediksi per sample
- Download hasil sebagai CSV

## Input Format

### ARFF Format
File ARFF standard UCR/UEA dengan kolom target (nama spesies)

### CSV Format
- 600+ kolom: time-series signal (wingbeat)
- Optional kolom terakhir: ground truth label

## Output

CSV file dengan kolom:
- `Index`: nomor sample
- `Prediksi`: hasil klasifikasi
- `Ground Truth`: label asli (jika ada)
