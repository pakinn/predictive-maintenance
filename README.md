# Predictive Maintenance — Vibration Analysis Pipeline (Sol1)

ระบบวิเคราะห์สัญญาณ Vibration เพื่อทำนายสภาพเครื่องจักรล่วงหน้า โดยใช้ Signal Processing + Machine Learning ตรวจจับความผิดปกติ (Anomaly Detection) จาก raw waveform ที่ได้จากเซ็นเซอร์วัดแรงสั่นสะเทือน

---

## วัตถุประสงค์

ระบบนี้ใช้สำหรับ:
- **ประเมินสุขภาพเครื่องจักร** (Health Assessment) โดยเปรียบเทียบค่า Feature กับ Baseline ครั้งแรกของแต่ละ Asset
- **ตรวจจับ Anomaly ด้วย AI** (One-Class SVM) บนชุด Feature ที่สกัดจากสัญญาณ Vibration
- **ติดตาม Trend** ของ Feature สำคัญตามเวลา เพื่อช่วยวางแผนซ่อมบำรุงเชิงรุก

**Assets ที่รองรับ:**
- Compressor (CH-06 NAA)
- Cooling Pump (OAH-02)
- Jockey Pump (M1A, M2A)

---

## วิธีการที่ใช้

### 1. Feature Engineering (`feature_engineering.py`)
สกัด Features 3 กลุ่มจาก raw waveform (`.txt`) ของแต่ละไฟล์:

| กลุ่ม Feature | รายการ | คำอธิบาย |
|---|---|---|
| **Wavelet** | `wavelet_energy_L0–L4`, `wavelet_entropy_L0–L4` | พลังงานและ Entropy ในแต่ละ Frequency Band (Daubechies-4, 4 levels) |
| **Envelope** | `env_rms`, `env_peak`, `env_dom_freq_hz`, `env_kurtosis`, `env_crest_factor` | วิเคราะห์จาก Hilbert Transform ตรวจจับ Bearing/Impact fault |
| **Statistical** | `rms_g`, `peak_abs_g`, `crest_factor`, `kurtosis`, `skewness`, `shape_factor`, `impulse_factor`, `zero_cross_rate` | Statistics ในโดเมนเวลา |

### 2. Health Assessment — Rule-Based (`analyze.py`)
- เปรียบเทียบแต่ละ Feature กับ **Baseline** (ไฟล์แรกสุดของ Asset)
- คำนวณ **Risk Score (0–100)** จากค่าเฉลี่ยของ % change ทั้งหมด
- แบ่ง Risk Level: **High** (≥ 60) / **Medium** (≥ 30) / **Low** (< 30)

### 3. AI Anomaly Detection — One-Class SVM (`ai_anomaly.py`)
- Train บน Feature ทั้งหมดของทุก Asset พร้อมกัน (Unsupervised)
- ใช้ **StandardScaler + One-Class SVM** (RBF kernel, ν = 0.15)
- คำนวณ **Anomaly Score (0–100)** จาก Decision Function
- ระบุ **Top Driver** ที่มีค่า Z-score สูงที่สุด (Feature ที่ผิดปกติมากที่สุด)

---

## วิธีรัน

### ติดตั้ง Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn PyWavelets scipy
```

### รันโปรแกรม
```bash
python analyze.py
```

> ⚠️ ข้อมูล Raw `.txt` ต้องอยู่ที่ `../Data/` (โฟลเดอร์ `Data` ถัดจาก `Sol1`)

---

## โครงสร้างไฟล์

```
Sol1/
├── analyze.py              # 🚀 Main script — รันตรงนี้
├── ai_anomaly.py           # One-Class SVM anomaly detection
├── feature_engineering.py  # สกัด Wavelet / Envelope / Stat features
├── load_data.py            # Parse raw waveform .txt
├── config.py               # ตั้งค่า parameters ทั้งหมด
└── outputs/
    ├── features.csv            # Feature ทุกไฟล์
    ├── health_assessment.csv   # ⭐ Rule-based risk score ต่อ Asset/Date
    ├── ai_assessment.csv       # ⭐ AI anomaly score ต่อ Asset/Date
    └── plots/
        ├── svm_anomaly_overview.png    # ⭐ ภาพรวม Anomaly (PCA + Bar)
        ├── svm_score_detail.png        # ⭐ Score + Top Driver ทุก Sample
        └── wavelet_envelope_trend_*.png  # Trend ของแต่ละ Asset
```

---

## Outputs ที่ต้องดูเป็นหลัก

### 🔴 ด่วนที่สุด — ไฟล์ที่ต้องดูก่อน

| Output | ไฟล์ | ดูอะไร |
|---|---|---|
| **AI Anomaly Ranking** | `outputs/ai_assessment.csv` | Column `anomaly_score_0_100` และ `anomaly_flag` — Asset ที่คะแนนสูงและ Flag เป็น `Anomaly` คือจุดที่น่ากังวลที่สุด |
| **Health Risk** | `outputs/health_assessment.csv` | Column `risk_score_0_100` และ `risk_level` — ดู `High` ก่อน พร้อม `largest_change_feature` บอกว่าอะไรเปลี่ยนมากที่สุด |
| **Anomaly Overview Plot** | `outputs/plots/svm_anomaly_overview.png` | กราฟ PCA (ซ้าย) แสดง clustering ของ Normal/Anomaly + Bar chart (ขวา) เรียง Score สูง-ต่ำ |
| **Score Detail Plot** | `outputs/plots/svm_score_detail.png` | Bar chart แนวนอน ดู Score และ Top Driver ของแต่ละ Sample ในมุมมองเดียว |

### 🟡 ดูเพิ่มเติม — ติดตาม Trend

| Output | ไฟล์ | ดูอะไร |
|---|---|---|
| **Trend per Asset** | `outputs/plots/wavelet_envelope_trend_*.png` | กราฟ Time-series ของ Wavelet Energy/Entropy และ Envelope features แยกตาม Asset — ดูว่า Feature ใดมีแนวโน้มสูงขึ้นต่อเนื่อง |
| **Raw Features** | `outputs/features.csv` | ข้อมูล Feature ดิบ ใช้สำหรับวิเคราะห์เพิ่มเติม |

---

## การตีความผล

```
anomaly_score_0_100
   0–49   → ปกติ (Normal)
  50–100  → ผิดปกติ (Anomaly) — ยิ่งสูงยิ่งน่ากังวล

risk_score_0_100
   < 30   → Low   — ยังดีอยู่
  30–59   → Medium — ควรติดตาม
  ≥ 60    → High  — ควรวางแผนซ่อมบำรุง

top_ai_driver    — Feature ที่ส่งผลต่อ Anomaly มากที่สุด
largest_change_feature — Feature ที่เปลี่ยนแปลงจาก Baseline มากที่สุด
```

---

## ปรับ Parameters (`config.py`)

| Parameter | ค่าปัจจุบัน | ความหมาย |
|---|---|---|
| `WAVELET` | `'db4'` | Wavelet family (Daubechies-4) |
| `WAVELET_LEVEL` | `4` | จำนวน Decomposition levels |
| `SVM_NU` | `0.15` | สัดส่วน Anomaly ที่คาดไว้ (~15%) |
| `SVM_GAMMA` | `'scale'` | Kernel width |
| `RISK_HIGH` | `60` | เกณฑ์ Risk Score → High |
| `RISK_MEDIUM` | `30` | เกณฑ์ Risk Score → Medium |
