from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import ISO_GROUP

from feature_extractor import build_feature_table
from svm_detector import build_ai_assessment, save_ai_plots
from config import RISK_HIGH, RISK_MEDIUM, HEALTH_SCALE, get_all_model_features

COMPARE_COLS = get_all_model_features()

def build_health_assessment(features: pd.DataFrame) -> pd.DataFrame:
    compare = [c for c in COMPARE_COLS if c in features.columns]
    rows = []

    for asset_id, group in features.groupby('asset_id'):
        group    = group.sort_values('analysis_date').reset_index(drop=True)
        baseline = group.iloc[0]

        for _, row in group.iterrows():
            rel_changes = []
            out = {
                'asset_id':      asset_id,
                'analysis_date': row['analysis_date'],
                'file':          row['file'],
                'baseline_file': baseline['file'],
            }
            for col in compare:
                base = baseline[col]
                rel  = (row[col] - base) / abs(base) if base != 0 else 0.0
                out[f'{col}_change_pct'] = rel * 100
                rel_changes.append(abs(rel))

            score = min(100.0, float(np.mean(rel_changes)) * HEALTH_SCALE)
            out['risk_score_0_100'] = round(score, 2)
            out['risk_level']       = (
                'High'   if score >= RISK_HIGH   else
                'Medium' if score >= RISK_MEDIUM else
                'Low'
            )
            out['largest_change_feature'] = max(
                compare,
                key=lambda c: abs(
                    (row[c] - baseline[c]) / (abs(baseline[c]) if baseline[c] != 0 else 1)
                ),
            )
            rows.append(out)

    return (pd.DataFrame(rows)
              .sort_values(['asset_id', 'analysis_date'])
              .reset_index(drop=True))

def build_iso_assessment(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in features.iterrows():
        asset = row['asset_id']
        thresholds = ISO_GROUP.get(asset, 
                        {'zone_b': 2.3, 'zone_c': 4.5, 'zone_d': 7.1})
        
        v = row.get('velocity_rms_mms', None)
        if v is None:
            zone = 'Unknown'
        elif v > thresholds['zone_d']:
            zone = 'D'   
        elif v > thresholds['zone_c']:
            zone = 'C'   
        elif v > thresholds['zone_b']:
            zone = 'B'   
        else:
            zone = 'A'   

        rows.append({
            'asset_id':        asset,
            'analysis_date':   row['analysis_date'],
            'file':            row['file'],
            'velocity_rms_mms': round(v, 3) if v else None,
            'iso_zone':        zone,
            'iso_standard':    'ISO 10816-3 Group 1',
        })

    return pd.DataFrame(rows).sort_values(['asset_id', 'analysis_date'])

PLOT_METRICS_WAVELET  = ['wavelet_energy_L1', 'wavelet_energy_L2', 'wavelet_entropy_L1']
PLOT_METRICS_ENVELOPE = ['env_rms', 'env_dom_freq_hz', 'env_kurtosis']

def save_trend_plots(features: pd.DataFrame, iso_assessment: pd.DataFrame,
                     plot_dir: str | Path) -> None:
    from config import ISO_ZONE_COLORS, ISO_GROUP

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    merged = features.merge(
        iso_assessment[['file', 'iso_zone']], 
        on='file', how='left'
    )

    for asset_id, group in merged.groupby('asset_id'):
        group = group.sort_values('analysis_date').reset_index(drop=True)
        x     = pd.to_datetime(group['analysis_date'])
        
        point_colors = [ISO_ZONE_COLORS.get(z, '#888') for z in group['iso_zone']]

        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        fig.suptitle(f'Vibration trend — {asset_id}', fontsize=11)

        ax = axes[0]
        ax.plot(x, group['velocity_rms_mms'], color='#555', linewidth=1.5, zorder=2)
        for i, (xi, yi, ci) in enumerate(zip(x, group['velocity_rms_mms'], point_colors)):
            ax.scatter(xi, yi, color=ci, s=80, zorder=3)

        thresholds = ISO_GROUP.get(asset_id, {'zone_b': 2.3, 'zone_c': 4.5, 'zone_d': 7.1})
        ymax = max(group['velocity_rms_mms'].max() * 1.3, thresholds['zone_d'] * 1.2)
        ax.axhspan(0,                      thresholds['zone_b'], alpha=0.08, color='#4CAF50', label='Zone A')
        ax.axhspan(thresholds['zone_b'],   thresholds['zone_c'], alpha=0.08, color='#2196F3', label='Zone B')
        ax.axhspan(thresholds['zone_c'],   thresholds['zone_d'], alpha=0.08, color='#FF9800', label='Zone C')
        ax.axhspan(thresholds['zone_d'],   ymax,                 alpha=0.08, color='#E53935', label='Zone D')

        ax.axhline(thresholds['zone_b'], color='#2196F3', linestyle='--', linewidth=0.8)
        ax.axhline(thresholds['zone_c'], color='#FF9800', linestyle='--', linewidth=0.8)
        ax.axhline(thresholds['zone_d'], color='#E53935', linestyle='--', linewidth=0.8)

        ax.set_title('Velocity RMS (mm/s) — ISO 10816-3', fontsize=9)
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=7, loc='upper left')
        ax.tick_params(axis='x', rotation=20, labelsize=8)

        for i, col in enumerate(['wavelet_energy_L1', 'env_rms', 'kurtosis'], start=1):
            if col not in group.columns:
                axes[i].set_visible(False)
                continue
            axes[i].plot(x, group[col], color='#5C6BC0', linewidth=1.5)
            for xi, yi, ci in zip(x, group[col], point_colors):
                axes[i].scatter(xi, yi, color=ci, s=60, zorder=3)
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(axis='x', rotation=20, labelsize=8)

        fig.tight_layout()
        safe_id = asset_id.replace('/', '_')
        fig.savefig(plot_dir / f'iso_trend_{safe_id}.png', dpi=150)
        plt.close(fig)

    print(f'[analyze] Saved trend plots → wavelet_envelope_trend_{{asset}}.png')

def print_summary(features: pd.DataFrame, assessment: pd.DataFrame,
                  ai_assessment: pd.DataFrame) -> None:
    print('\n' + '=' * 55)
    print('PIPELINE COMPLETE')
    print('=' * 55)
    print(f'  Files processed  : {len(features)}')
    print(f'  Assets found     : {features["asset_id"].nunique()}')
    print(f'  Features per file: {len([c for c in features.columns if c not in ["file","asset_id","analysis_date","equipment_header","meas_point","header_datetime","amplitude_unit","n_samples","duration_ms","sample_interval_ms","sampling_rate_hz"]])}')
    print()
    print('  Risk distribution:')
    for level in ['High', 'Medium', 'Low']:
        n = (assessment['risk_level'] == level).sum()
        print(f'    {level:6s}: {n}')
    print()
    print('  AI anomaly result:')
    for flag in ['Anomaly', 'Normal']:
        n = (ai_assessment['anomaly_flag'] == flag).sum()
        print(f'    {flag:8s}: {n}')
    print('=' * 55 + '\n')

def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root.parent / 'Data' 
    features = build_feature_table(data_dir)

    print('[1/5] Loading and extracting features...')
    features = build_feature_table(data_dir)

    print('[2/5] Building health assessment (rule-based)...')
    assessment = build_health_assessment(features)

    print('[3/5] Running One-Class SVM anomaly detection...')
    ai_assessment = build_ai_assessment(features)

    iso_assessment = build_iso_assessment(features)

    print('[4/5] Saving outputs...')
    out_dir = root / 'outputs'
    out_dir.mkdir(exist_ok=True)
    features.to_csv(out_dir / 'features.csv', index=False)
    assessment.to_csv(out_dir / 'health_assessment.csv', index=False)
    ai_assessment.to_csv(out_dir / 'ai_assessment.csv', index=False)

    print('[5/5] Generating plots...')
    save_trend_plots(features, iso_assessment, out_dir / 'plots')
    save_ai_plots(features, ai_assessment, iso_assessment, out_dir / 'plots')

    print_summary(features, assessment, ai_assessment)

if __name__ == '__main__':
    main()