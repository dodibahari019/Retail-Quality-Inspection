---
title: Smart Retail Quality Inspection - RF-DETR-Nano
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# Smart Retail Quality Inspection — RF-DETR-Nano

Demo deteksi kerusakan botol minuman (**Dent**, **Label Damage**, **Missing Cap**,
**Seal Damage**) memakai model **RF-DETR-Nano**.

Pipeline inferensi di `app.py` ini konsisten dengan:
- Tahap Modeling (`modeling_v6_rfdetr_nano_single.py`) — training checkpoint
- Tahap Evaluasi (`evaluasi_final_v1_rfdetr_nano_test.py`) — pengukuran performa
  pada test set independen

Tidak ada logic deteksi (nama/urutan kelas, skema warna, cara load model) yang
diubah di app ini — file ini murni membungkus checkpoint yang sudah dilatih ke
antarmuka Streamlit untuk demo publik.

## Struktur file yang wajib ada di Space ini

```
.
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── best.pth          
│   └── metadata.json     
└── examples/              
    ├── contoh1.jpg
    └── contoh2.jpg
```

Lihat langkah upload lengkap di percakapan/dokumentasi deploy.
