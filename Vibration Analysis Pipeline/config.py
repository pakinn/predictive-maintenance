# Wavelet
WAVELET        = 'db4'   # Daubechies-4 เหมาะกับ vibration signal
WAVELET_LEVEL  = 4       # จำนวน decomposition levels (L0=approx, L1-L4=detail)

# One-Class SVM 
SVM_NU         = 0.15    # สัดส่วน anomaly ที่คาดไว้ (0.0–1.0)  ~15% ของ 9 ไฟล์ ≈ 1-2 ไฟล์
SVM_KERNEL     = 'rbf'   # rbf ดีที่สุดสำหรับข้อมูลไม่ linear
SVM_GAMMA      = 'scale' # auto-scale gamma ตาม feature variance

# Health assessment thresholds 
RISK_HIGH      = 60      # risk_score >= 60 → High
RISK_MEDIUM    = 30      # risk_score >= 30 → Medium (else Low)
HEALTH_SCALE   = 170     # scaling factor สำหรับ heuristic risk score

STAT_FEATURE_NAMES = [
    'rms_g', 'peak_abs_g', 'crest_factor', 'kurtosis',
    'skewness', 'shape_factor', 'impulse_factor', 'zero_cross_rate',
]
ENVELOPE_FEATURE_NAMES = [
    'env_rms', 'env_peak', 'env_dom_freq_hz', 'env_kurtosis', 'env_crest_factor',
]

def get_wavelet_feature_names(level: int = WAVELET_LEVEL) -> list[str]:
    names = []
    for i in range(level + 1):
        names.append(f'wavelet_energy_L{i}')
        names.append(f'wavelet_entropy_L{i}')
    return names

def get_all_model_features(level: int = WAVELET_LEVEL) -> list[str]:
    return get_wavelet_feature_names(level) + ENVELOPE_FEATURE_NAMES + STAT_FEATURE_NAMES