from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import (
    SVM_NU, SVM_KERNEL, SVM_GAMMA,
    get_all_model_features,
)

MODEL_FEATURES = get_all_model_features()

# Core Assessment
def build_ai_assessment(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy().reset_index(drop=True)

    available = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[available].astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = OneClassSVM(nu=SVM_NU, kernel=SVM_KERNEL, gamma=SVM_GAMMA)
    preds = model.fit_predict(Xs)         
    raw_scores = -model.decision_function(Xs)  

    min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
    if max_s - min_s < 1e-9:
        norm_scores = np.zeros(len(raw_scores))
    else:
        norm_scores = (raw_scores - min_s) / (max_s - min_s)

    z = np.abs(Xs)
    top_idx  = z.argmax(axis=1)
    top_feat = [available[i] for i in top_idx]
    top_z    = z[np.arange(len(df)), top_idx]

    df['anomaly_score_0_100']  = np.round(norm_scores * 100, 2)
    df['anomaly_flag']         = np.where(preds == -1, 'Anomaly', 'Normal')
    df['model_name']           = 'OneClassSVM'
    df['top_ai_driver']        = top_feat
    df['top_ai_driver_zscore'] = np.round(top_z, 3)

    return df[[
        'asset_id', 'analysis_date', 'file', 'model_name',
        'anomaly_score_0_100', 'anomaly_flag',
        'top_ai_driver', 'top_ai_driver_zscore',
    ]].sort_values(['anomaly_score_0_100', 'analysis_date'], ascending=[False, True])

# Plots
def save_ai_plots(features: pd.DataFrame, ai_assessment: pd.DataFrame,
                  plot_dir: str | Path) -> None:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    merged = features.merge(
        ai_assessment[['file', 'anomaly_score_0_100', 'anomaly_flag', 'top_ai_driver']],
        on='file', how='left',
    )

    available = [f for f in MODEL_FEATURES if f in merged.columns]
    Xs  = StandardScaler().fit_transform(merged[available].astype(float))
    pts = PCA(n_components=2, random_state=42).fit_transform(Xs)

    colors  = {'Normal': '#2196F3', 'Anomaly': '#E53935'}
    markers = {'Normal': 'o', 'Anomaly': 'X'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # PCA
    for flag in ['Normal', 'Anomaly']:
        mask = merged['anomaly_flag'] == flag
        ax1.scatter(pts[mask, 0], pts[mask, 1],
                    marker=markers[flag], color=colors[flag],
                    s=90, label=flag, zorder=3)
    for i, row in merged.iterrows():
        lbl = f"{row['asset_id']}\n{str(row['analysis_date'])[:10]}"
        ax1.annotate(lbl, (pts[i, 0], pts[i, 1]),
                     fontsize=7, alpha=0.8,
                     xytext=(4, 4), textcoords='offset points')
    ax1.set_title('One-Class SVM — PCA projection')
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
    ax1.legend()

    # Score bar
    sp = merged.sort_values('anomaly_score_0_100', ascending=False).reset_index(drop=True)
    bar_colors = [colors.get(f, '#888') for f in sp['anomaly_flag']]
    xlabels    = [f"{a}\n{str(d)[:10]}"
                  for a, d in zip(sp['asset_id'], sp['analysis_date'])]
    ax2.bar(range(len(sp)), sp['anomaly_score_0_100'],
            color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='Score=50')
    ax2.set_xticks(range(len(sp)))
    ax2.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('Anomaly score (0–100)')
    ax2.set_title('Anomaly ranking')
    ax2.legend()

    fig.suptitle('Vibration Anomaly Detection — One-Class SVM', fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(plot_dir / 'svm_anomaly_overview.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(sp)), sp['anomaly_score_0_100'],
                   color=[colors.get(f, '#888') for f in sp['anomaly_flag']])
    for i, (_, row) in enumerate(sp.iterrows()):
        ax.text(row['anomaly_score_0_100'] + 0.8, i,
                f"driver: {row['top_ai_driver']}",
                va='center', fontsize=8, color='#444')
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels([f"{a} | {str(d)[:10]}"
                        for a, d in zip(sp['asset_id'], sp['analysis_date'])],
                       fontsize=8)
    ax.axvline(x=50, color='orange', linestyle='--', linewidth=1)
    ax.set_xlabel('Anomaly score (0–100)')
    ax.set_title('Score & top anomaly driver per sample')
    fig.tight_layout()
    fig.savefig(plot_dir / 'svm_score_detail.png', dpi=160)
    plt.close(fig)

    print(f'[ai_anomaly] Saved: svm_anomaly_overview.png, svm_score_detail.png')