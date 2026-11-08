# Gambar dan sumber Bab 2

Sumber yang diterapkan langsung pada Overleaf: `02-bab2.tex` dan penambahan rujukan pada `pustaka.bib`. PNG diunggah ke `assets/pics/` proyek yang sama.

Bangun ulang delapan gambar dari root repository:

```sh
.venv/bin/python scripts/figures/build_bab2_figures.py
```

Script hanya membaca dataset dan notebook; tidak melatih atau mengubah model. PDF gambar merupakan versi vektor untuk ekspor; versi PNG dipakai pada Overleaf. `provenance.json` mencatat sumber, window yang dipilih, batas waktu split, dan urutan subset node.

- Window: koridor Cawang dari data pelatihan; titik asli dan imputasi dibedakan. Target aktual bukan prediksi. Contoh dipilih untuk memperjelas variasi, bukan mengukur performa.
- Graf: subgraf terinduksi dari adjacency 64 node tersimpan. Koordinat asli tanpa peta dasar; panah bukan arah jalan fisik.
- Arsitektur, dilasi, embedding, konfigurasi: skema operasi berdasarkan kode notebook dan konsep Graph WaveNet.
- Pembagian waktu: rentang awal–akhir berkas praproses, dalam UTC; tidak menyatakan semua jam tersedia.
- Subset: urutan persis NODE_ORDER pada notebook scalability, tanpa menjalankan notebook.

Lingkup revisi: Bab 2. Error gambar lama Bab 3 tidak diubah dalam revisi ini.
