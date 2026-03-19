# Predictive Maintenance — Vibration Analysis Pipeline

ระบบวิเคราะห์สัญญาณ Vibration เพื่อทำนายสภาพเครื่องจักรล่วงหน้า โดยใช้ Signal Processing + Machine Learning ตรวจจับความผิดปกติ (Anomaly Detection) จาก raw waveform ที่ได้จากเซ็นเซอร์วัดแรงสั่นสะเทือน

---

## วัตถุประสงค์

ระบบนี้ใช้สำหรับ:
- **ประเมินสุขภาพเครื่องจักร** (Health Assessment) โดยเปรียบเทียบค่า Feature กับ Baseline ครั้งแรกของแต่ละ Asset
- **ตรวจสอบตามมาตรฐาน ISO 10816-3** โดยคำนวณ Velocity RMS (mm/s) และจัดกลุ่ม Zone A–D
- **ตรวจจับ Anomaly ด้วย AI** (One-Class SVM) บนชุด Feature ที่สกัดจากสัญญาณ Vibration
- **ติดตาม Trend** ของ Feature สำคัญตามเวลา เพื่อช่วยวางแผนซ่อมบำรุงเชิงรุก

**Assets ที่รองรับ:**
- Compressor (CH-06 NAA) → `compressor_ch06_naa`
- Cooling Pump (OAH-02) → `cooling_pump_oah02_m1h`
- Jockey Pump M1A → `jockey_pump_m1a`
- Jockey Pump M2A → `jockey_pump_m2a`

---

## วิธีการที่ใช้

### 1. Feature Engineering (`feature_extractor.py`)
สกัด Features 4 กลุ่มจาก raw waveform (`.txt`) ของแต่ละไฟล์:

| กลุ่ม Feature | รายการ | คำอธิบาย |
|---|---|---|
| **Wavelet** | `wavelet_energy_L0–L4`, `wavelet_entropy_L0–L4` | พลังงานและ Entropy ในแต่ละ Frequency Band (Daubechies-4, 4 levels) |
| **Envelope** | `env_rms`, `env_peak`, `env_dom_freq_hz`, `env_kurtosis`, `env_crest_factor` | วิเคราะห์จาก Hilbert Transform ตรวจจับ Bearing/Impact fault |
| **Statistical** | `rms_g`, `peak_abs_g`, `crest_factor`, `kurtosis`, `skewness`, `shape_factor`, `impulse_factor`, `zero_cross_rate` | Statistics ในโดเมนเวลา |
| **Velocity RMS** | `velocity_rms_mms` | แปลง Acceleration → Velocity ด้วย Cumulative Trapezoid Integration (หน่วย mm/s) |

### 2. Health Assessment — Rule-Based (`main.py`)
- เปรียบเทียบแต่ละ Feature กับ **Baseline** (ไฟล์แรกสุดของ Asset)
- คำนวณ **Risk Score (0–100)** จากค่าเฉลี่ยของ % change ทั้งหมด (scaling factor = 170)
- แบ่ง Risk Level: **High** (≥ 60) / **Medium** (≥ 30) / **Low** (< 30)

### 3. ISO 10816-3 Assessment (`main.py`)
- คำนวณ Velocity RMS (mm/s) จากสัญญาณ Acceleration
- จัดกลุ่มตามมาตรฐาน **ISO 10816-3 Group 1**:

| Zone | ความหมาย | เกณฑ์ (mm/s) |
|---|---|---|
| **A** | ปกติ (เครื่องจักรใหม่/ดี) | < 2.3 |
| **B** | ยอมรับได้สำหรับการใช้งานระยะยาว | 2.3 – 4.5 |
| **C** | เริ่มน่ากังวล ควรวางแผนซ่อม | 4.5 – 7.1 |
| **D** | อันตราย ต้องหยุดซ่อมทันที | > 7.1 |

### 4. AI Anomaly Detection — One-Class SVM (`svm_detector.py`)
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
python main.py
```

> ⚠️ ข้อมูล Raw `.txt` ต้องอยู่ที่ `../Data/` (โฟลเดอร์ `Data` ถัดจาก `Vibration Analysis Pipeline`)

---

## โครงสร้างไฟล์

```
Vibration Analysis Pipeline/
├── main.py                 # 🚀 Main script — รันตรงนี้
├── svm_detector.py         # One-Class SVM anomaly detection + plots
├── feature_extractor.py    # สกัด Wavelet / Envelope / Stat / Velocity features
├── data_parser.py          # Parse raw waveform .txt
├── config.py               # ตั้งค่า parameters ทั้งหมด
└── outputs/
    ├── features.csv            # Feature ทุกไฟล์
    ├── health_assessment.csv   # ⭐ Rule-based risk score ต่อ Asset/Date
    ├── ai_assessment.csv       # ⭐ AI anomaly score ต่อ Asset/Date
    └── plots/
        ├── svm_anomaly_overview.png    # ⭐ ภาพรวม Anomaly (PCA + Bar)
        ├── svm_score_detail.png        # ⭐ Score + Top Driver ทุก Sample
        └── iso_trend_*.png             # Trend ของแต่ละ Asset (ISO Zone color-coded)
```

---

## Outputs ที่ต้องดูเป็นหลัก

### 🔴 ด่วนที่สุด — ไฟล์ที่ต้องดูก่อน

| Output | ไฟล์ | ดูอะไร |
|---|---|---|
| **AI Anomaly Ranking** | `outputs/ai_assessment.csv` | Column `anomaly_score_0_100` และ `anomaly_flag` — Asset ที่คะแนนสูงและ Flag เป็น `Anomaly` คือจุดที่น่ากังวลที่สุด |
| **Health Risk** | `outputs/health_assessment.csv` | Column `risk_score_0_100` และ `risk_level` — ดู `High` ก่อน พร้อม `largest_change_feature` บอกว่าอะไรเปลี่ยนมากที่สุด |
| **ISO Zone** | `outputs/ai_assessment.csv` (รวมใน plot) | ดูสีของจุดในกราฟ — เขียว=A, น้ำเงิน=B, ส้ม=C, แดง=D |
| **Anomaly Overview Plot** | `outputs/plots/svm_anomaly_overview.png` | กราฟ PCA (ซ้าย) แสดง clustering พร้อม ISO Zone color + Bar chart (ขวา) เรียง Score สูง-ต่ำ |
| **Score Detail Plot** | `outputs/plots/svm_score_detail.png` | Bar chart แนวนอน ดู Score และ Top Driver ของแต่ละ Sample ในมุมมองเดียว |

### 🟡 ดูเพิ่มเติม — ติดตาม Trend

| Output | ไฟล์ | ดูอะไร |
|---|---|---|
| **Trend per Asset** | `outputs/plots/iso_trend_*.png` | กราฟ Time-series ของ Velocity RMS + Wavelet Energy + Envelope + Kurtosis แยกตาม Asset พร้อม ISO Zone threshold lines |
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

iso_zone
  A → velocity_rms_mms < 2.3    — ปกติดี
  B → 2.3 ≤ velocity_rms_mms < 4.5 — ยอมรับได้
  C → 4.5 ≤ velocity_rms_mms < 7.1 — ควรวางแผนซ่อม
  D → velocity_rms_mms ≥ 7.1    — อันตราย

top_ai_driver           — Feature ที่ส่งผลต่อ Anomaly มากที่สุด
largest_change_feature  — Feature ที่เปลี่ยนแปลงจาก Baseline มากที่สุด
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
| `HEALTH_SCALE` | `170` | Scaling factor สำหรับ heuristic risk score |
| `ISO_ZONE_B/C/D` | `2.3 / 4.5 / 7.1` | เกณฑ์ ISO 10816-3 Zone (mm/s) |
