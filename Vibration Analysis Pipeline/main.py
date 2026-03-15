from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


PLOT_METRICS_WAVELET  = ['wavelet_energy_L1', 'wavelet_energy_L2', 'wavelet_entropy_L1']
PLOT_METRICS_ENVELOPE = ['env_rms', 'env_dom_freq_hz', 'env_kurtosis']

def save_trend_plots(features: pd.DataFrame, plot_dir: str | Path) -> None:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    w_cols = [c for c in PLOT_METRICS_WAVELET  if c in features.columns]
    e_cols = [c for c in PLOT_METRICS_ENVELOPE if c in features.columns]
    n_rows = max(len(w_cols), len(e_cols))

    if n_rows == 0:
        return

    for asset_id, group in features.groupby('asset_id'):
        group = group.sort_values('analysis_date').reset_index(drop=True)
        x     = pd.to_datetime(group['analysis_date'])

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows), squeeze=False)
        fig.suptitle(f'Vibration trend — {asset_id}', fontsize=11)

        for i, col in enumerate(w_cols):
            axes[i][0].plot(x, group[col], marker='o', color='#5C6BC0', linewidth=1.5)
            axes[i][0].set_title(col, fontsize=9)
            axes[i][0].tick_params(axis='x', rotation=20, labelsize=8)

        for i, col in enumerate(e_cols):
            axes[i][1].plot(x, group[col], marker='s', color='#26A69A', linewidth=1.5)
            axes[i][1].set_title(col, fontsize=9)
            axes[i][1].tick_params(axis='x', rotation=20, labelsize=8)

        for i in range(len(w_cols), n_rows):
            axes[i][0].set_visible(False)
        for i in range(len(e_cols), n_rows):
            axes[i][1].set_visible(False)

        fig.tight_layout()
        safe_id = asset_id.replace('/', '_')
        fig.savefig(plot_dir / f'wavelet_envelope_trend_{safe_id}.png', dpi=150)
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

    print('[4/5] Saving outputs...')
    out_dir = root / 'outputs'
    out_dir.mkdir(exist_ok=True)
    features.to_csv(out_dir / 'features.csv', index=False)
    assessment.to_csv(out_dir / 'health_assessment.csv', index=False)
    ai_assessment.to_csv(out_dir / 'ai_assessment.csv', index=False)

    print('[5/5] Generating plots...')
    save_trend_plots(features, out_dir / 'plots')
    save_ai_plots(features, ai_assessment, out_dir / 'plots')

    print_summary(features, assessment, ai_assessment)

if __name__ == '__main__':
    main()