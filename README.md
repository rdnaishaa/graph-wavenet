# Prediksi Kecepatan Lalu Lintas — Final64

Skripsi tentang **Fixed, Adaptive, dan Hybrid Graph WaveNet**, dengan GC-LSTM-style sebagai pembanding pendukung. Masukan model adalah 12 jam kecepatan pada 64 koridor; target prediksi adalah +1, +3, dan +6 jam.

## Mulai dari notebook

**Notebook utama: [`notebooks/graphwavenet_final64_audited.ipynb`](notebooks/graphwavenet_final64_audited.ipynb).** Gunakan kernel Python dari `.venv`. Tidak ada script training alternatif yang perlu dipilih.

| Notebook | Kegunaan |
|---|---|
| `graphwavenet_final64_audited.ipynb` | Training dan evaluasi tiga konfigurasi GWN, lima seed |
| `gclstm_final64_baseline.ipynb` | Baseline dengan normalisasi masked loss terkoreksi |
| `model_comparison_final64.ipynb` | Ringkasan hasil per horizon dan statistik antar-seed |
| `training_cost_benchmark_final64.ipynb` | Pengukuran biaya training GWN |
| `scalability_hybrid_vs_gclstm.ipynb` | Eksperimen subset nested 8, 16, 32, dan 64 node |
| `graphwavenet_final_analysis_demo.ipynb` | Analisis dan demo GWN dari hasil tersimpan |
| `gwn_vs_gclstm_demo.ipynb` | Demo perbandingan GWN dan baseline |
| `live_forecast_final64.ipynb` | Inferensi dari 12 snapshot per jam yang lengkap |
| `gambar_seminar.ipynb` | Gambar pendukung seminar |

Cell training menulis bobot dan hasil eksperimen. Untuk menampilkan hasil yang sudah ada, gunakan notebook analisis/demo atau dashboard; tidak perlu menjalankan ulang training.

## Instalasi dan dashboard

Gunakan Python 3.11, dari root proyek:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name tomtom-final64 --display-name "TomTom Final64"
python -m streamlit run streamlit_app/app.py
```

Versi dependensi mengikuti lingkungan penelitian. `requirements.txt` berisi runtime; `requirements-dev.txt` menambahkan kernel notebook dan Ruff. Dashboard memakai bobot tersimpan, bukan melatih ulang model.

Untuk pengambilan data dan fitur rute TomTom, salin `.env.example` ke `.env`, lalu isi `TOMTOM_API_KEY`. Demo historis tidak memerlukan API key.

## Struktur yang dipakai

| Lokasi | Isi |
|---|---|
| `notebooks/` | Alur eksperimen utama, analisis, dan demo |
| `streamlit_app/` | Dashboard dan seluruh modul pendukungnya |
| `scripts/preprocessing/` | Audit Final64, pemilihan koridor, preprocessing, window, dan fixed graph |
| `scripts/evaluation/evaluate_final_baselines.py` | Evaluasi baseline statistik |
| `scripts/live/` | Koleksi TomTom dan pembentukan input live 12 jam |
| `scripts/figures/` | Gambar Bab 2 dan Bab 3 |
| `data/final_64/` | Dataset beku, mask, scaler, dan urutan node |
| `outputs/data_audit/` | Bukti audit dan pemilihan data penelitian |
| `outputs/final_64/` | Hasil eksperimen, bobot, adjacency, dan tabel |
| `outputs/thesis/` | Gambar dan sumber naskah lokal |
| `tests/` | Pemeriksaan inferensi, mask, loss, notebook, dan input live |

Audit Final64 dipertahankan. Script eksplorasi split 14/24/64 node, training STGNN pendahuluan, analisis yang hanya membaca model pendahuluan, serta notebook screening model di luar fokus telah dihapus dari alur kode. Hasil eksperimen historis tetap disimpan sebagai bukti penelitian; bukan alternatif model utama.

## Asal hasil dan batas interpretasi

- GWN dashboard memakai bobot seed 42 di `outputs/final_64/graphwavenet_audited/`.
- Graph Explorer menampilkan adjacency **Adaptive GWN seed 42**, bukan Hybrid atau STGNN pendahuluan.
- Hasil utama mencakup seed 42, 123, 456, 789, dan 2026.
- GC-LSTM historis di `gclstm_baseline/` memakai normalisasi loss berbeda. Perbandingannya masih deskriptif.
- Notebook baseline aktif menggunakan `sum(mask * squared_error) / sum(mask)`. Run baru ditulis ke `gclstm_baseline_corrected/`; hasil historis tidak ditimpa. Tabel perbandingan masih memakai hasil historis sampai run koreksi tersedia dan sumber analisis diperbarui.
- SD lintas horizon dihitung dari rata-rata horizon per seed, kemudian SD antar-seed.
- Scalability adalah eksperimen terpisah dengan training per ukuran graph; jangan digabung sebagai hasil lima seed eksperimen utama.
- Folder hasil `fixed_stgnn`, `adaptive_stgnn`, `gru`, `experiments`, dan `model_screening` adalah bukti eksperimen pendahuluan.

## Audit dan preprocessing

Data mentah tidak dibundel dalam proyek. Atur lokasi folder CSV sebelum menjalankan audit, preprocessing awal, atau gambar Bab 3:

```bash
export TOMTOM_RAW_DATA_DIR="/path/ke/full_data_update"
python scripts/preprocessing/analyze_network_size.py
python scripts/preprocessing/audit_candidate_64.py
```

Tanpa variabel tersebut, script mencari `data_update_dosen/full_data_update/` di root proyek. Audit menulis laporan ke `outputs/data_audit/`. `prepare_final_dataset.py` menggunakan daftar koridor audit yang dibekukan.

Untuk membangun ulang dataset dan graph secara sengaja:

```bash
python scripts/preprocessing/prepare_final_dataset.py
python scripts/preprocessing/build_final_stgnn_dataset.py
python scripts/preprocessing/build_final_fixed_graph.py
```

Perintah ini **menulis ulang artefak**. Dataset beku sudah tersedia; tidak perlu menjalankannya untuk demo. `corridors.csv` diperlukan untuk metadata geografis. Urutan preprocessing mengikuti implementasi penelitian, termasuk batasan seleksi koridor dari seluruh arsip dan penentuan gap menggunakan ketersediaan jam berikutnya.

## Membuat gambar skripsi

```bash
python scripts/figures/build_bab2_figures.py
# Memerlukan TOMTOM_RAW_DATA_DIR untuk audit terhadap CSV mentah:
python scripts/figures/build_bab3_preprocessing_figures.py
```

PNG/PDF ditulis ke `outputs/thesis/bab2/` dan `outputs/thesis/bab3/`. Script gambar Bab 3 memeriksa mask, imputasi, validitas waktu, dan scaler sebelum membuat gambar. Dataset dan bobot tidak diubah.

## Demo live

```bash
python scripts/live/collect_tomtom_live.py
python scripts/live/build_live_latest_12h.py
```

Kolektor mengambil **satu snapshot saat dijalankan**. Ulangi setiap jam sampai tersedia 12 snapshot berurutan untuk semua 64 node; kemudian buka `notebooks/live_forecast_final64.ipynb`. Koleksi dan state lokal disimpan di `data/collected/`, yang tidak dibundel pada repo baru.

`collect_hourly_session.py` adalah pembungkus sesi berbatas waktu. Deadline di `hourly_session.json` tidak boleh direset selama sesi. Jam terlewat tidak dapat diisi sebagai observasi asli. Dua belas snapshot per jam membentang 11 jam dari snapshot pertama. Snapshot sesaat berbeda dari rata-rata selama satu jam; gunakan untuk demo inferensi, bukan klaim evaluasi live tanpa observasi target yang sesuai.

## Pemeriksaan sebelum push

```bash
python -m unittest discover -s tests -v
ruff check streamlit_app scripts tests
ruff format --check streamlit_app scripts notebooks tests
```

Tes memakai artefak lokal, tanpa request TomTom atau training eksperimen. Satu batch sintetis hanya menguji kontrak masked loss Keras. Notebook perbandingan diuji di direktori sementara sehingga hasil penelitian tidak ditimpa.

Untuk repo baru, sertakan kode, notebook, `corridors.csv`, `data/final_64/`, dan artefak hasil yang diperlukan demo. Jangan salin `.git` lama jika memulai riwayat Git baru. `.env`, `.venv`, cache, data mentah, dan state koleksi lokal diabaikan oleh `.gitignore`; aturan ignore tidak otomatis menghapus file yang telah tracked pada repo lama. Tidak ada commit atau push otomatis.
