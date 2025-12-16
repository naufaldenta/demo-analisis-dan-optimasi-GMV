# Prediksi dan Optimasi GMV Iklan Menggunakan Linear Regression

Aplikasi Streamlit untuk:
1) Prediksi GMV (proxy: Sales) dari rencana budget iklan
2) Optimasi sederhana alokasi budget via simulasi skenario

## Dataset
Dataset Kaggle: Advertising Sales Dataset (kolom: TV, Radio, Newspaper, Sales)

> Catatan mapping untuk relevansi iklan digital:
- TV        : Meta Ads Spend
- Radio     : Google Ads Spend
- Newspaper : TikTok Ads Spend
- Sales     : proxy GMV

Simpan dataset sebagai:
`data/advertising.csv`

## Jalankan Lokal
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
