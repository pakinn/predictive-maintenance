from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert

from data_parser import parse_waveform_txt
from config import WAVELET, WAVELET_LEVEL

def extract_wavelet_features(amp: np.ndarray, wavelet: str = WAVELET,
                              level: int = WAVELET_LEVEL) -> dict:
    coeffs = pywt.wavedec(amp, wavelet, level=level)
    features = {}
    for i, c in enumerate(coeffs):
        energy = float(np.sum(c ** 2))
        p = c ** 2 / (energy + 1e-12)
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        features[f'wavelet_energy_L{i}'] = energy
        features[f'wavelet_entropy_L{i}'] = entropy
    return features

def extract_envelope_features(amp: np.ndarray, fs_hz: float) -> dict:
    envelope = np.abs(hilbert(amp))
    env_fft   = np.abs(np.fft.rfft(envelope - envelope.mean())) ** 2
    freqs     = np.fft.rfftfreq(len(envelope), d=1.0 / fs_hz)

    env_rms   = float(np.sqrt(np.mean(envelope ** 2)))
    env_peak  = float(envelope.max())

    dom_idx = int(np.argmax(env_fft[1:]) + 1)

    return {
        'env_rms':          env_rms,
        'env_peak':         env_peak,
        'env_dom_freq_hz':  float(freqs[dom_idx]),
        'env_kurtosis':     float(
            (((envelope - envelope.mean()) / (envelope.std(ddof=0) + 1e-12)) ** 4).mean()
        ),
        'env_crest_factor': float(env_peak / (env_rms + 1e-12)),
    }

def extract_stat_features(amp: np.ndarray, fs_hz: float) -> dict:
    rms       = float(np.sqrt(np.mean(amp ** 2)))
    peak      = float(np.max(np.abs(amp)))
    abs_mean  = float(np.mean(np.abs(amp)))
    std       = float(amp.std(ddof=0))
    centered  = amp - amp.mean()

    shape_factor   = rms / (abs_mean + 1e-12)         # RMS / |mean|
    impulse_factor = peak / (abs_mean + 1e-12)        # peak / |mean|
    skewness       = float((centered ** 3).mean() / (std ** 3 + 1e-12))
    kurtosis       = float((centered ** 4).mean() / (std ** 4 + 1e-12))
    crest_factor   = peak / (rms + 1e-12)
    zero_cross     = float(np.mean(np.diff(np.signbit(amp)) != 0))

    return {
        'rms_g':          rms,
        'peak_abs_g':     peak,
        'crest_factor':   crest_factor,
        'kurtosis':       kurtosis,
        'skewness':       skewness,
        'shape_factor':   shape_factor,
        'impulse_factor': impulse_factor,
        'zero_cross_rate': zero_cross,
    }

def extract_all_features(df: pd.DataFrame) -> dict:
    amp = df['amplitude_g'].to_numpy().copy()
    time_ms  = df['time_ms'].to_numpy()
    dt_ms    = float(np.median(np.diff(np.sort(np.unique(time_ms)))))
    fs_hz    = 1000.0 / dt_ms

    base = {
        'duration_ms':       float(time_ms.max() - time_ms.min()),
        'sample_interval_ms': dt_ms,
        'sampling_rate_hz':  fs_hz,
    }
    wavelet  = extract_wavelet_features(amp)
    envelope = extract_envelope_features(amp, fs_hz)
    stats    = extract_stat_features(amp, fs_hz)

    return {**base, **wavelet, **envelope, **stats}

def infer_asset_id(file_name: str, meas_point: str) -> str:
    if 'CH-06' in file_name:
        return 'compressor_ch06_naa'
    if 'Cooling Pump' in file_name:
        return 'cooling_pump_oah02_m1h'
    if 'Jockey pump' in file_name and '-M1A' in meas_point:
        return 'jockey_pump_m1a'
    if 'Jockey pump' in file_name and '-M2A' in meas_point:
        return 'jockey_pump_m2a'
    return file_name.lower().replace(' ', '_')

def infer_analysis_date(file_name: str, fallback: str | None) -> str | None:
    mapping = {
        'Jun24': '2024-06-28',
        'Sep24': '2024-09-04',
        'Oct24': '2024-10-16',
    }
    for key, value in mapping.items():
        if key in file_name:
            return value
    return fallback[:10] if fallback else None

def build_feature_table(raw_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(raw_dir).glob('*.txt')):
        meta, df = parse_waveform_txt(path)
        rows.append({
            **meta,
            'asset_id':      infer_asset_id(meta['file'], meta['meas_point'] or ''),
            'analysis_date': infer_analysis_date(meta['file'], meta['header_datetime']),
            **extract_all_features(df),
        })
    return pd.DataFrame(rows).sort_values(['asset_id', 'analysis_date'])