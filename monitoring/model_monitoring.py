"""
monitoring/model_monitoring.py
───────────────────────────────
Production-style model monitoring for a credit risk scoring model.

Computes, per monitoring window:
  - Population Stability Index (PSI): detects input/output drift
  - AUC and Gini coefficient: tracks model discrimination
  - KS statistic: score separation between classes
  - Expected Calibration Error (ECE): detects probability drift
  - Score distribution percentiles: early warning of distribution shift

Alert thresholds:
  PSI < 0.10     -> stable
  PSI 0.10-0.25  -> investigate
  PSI > 0.25     -> retrain

Usage:
  python monitoring/model_monitoring.py
  python monitoring/model_monitoring.py --windows 6 --seed 99

To use with your real model outputs:
  Replace the generate_synthetic_data() call with your own
  (model_scores, y_true, feature_matrix, window_labels) arrays.
"""

import argparse
import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Metric functions ───────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    Measures how much the distribution of `actual` has shifted from `expected`.
    Buckets are defined on the expected distribution's deciles so that each
    expected bucket has equal population - the standard approach.
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid log(0): clip to a small positive value
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    psi_value = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi_value, 5)


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error - measures probability drift."""
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="quantile"
    )
    ece = float(np.mean(np.abs(fraction_of_positives - mean_predicted)))
    return round(ece, 5)


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS statistic: maximum separation between default and non-default score CDFs."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    scores_pos = np.sort(y_score_arr[y_true_arr == 1])
    scores_neg = np.sort(y_score_arr[y_true_arr == 0])

    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return 0.0

    all_scores = np.sort(np.concatenate([scores_pos, scores_neg]))
    pos_cdf = np.searchsorted(scores_pos, all_scores, side="right") / len(scores_pos)
    neg_cdf = np.searchsorted(scores_neg, all_scores, side="right") / len(scores_neg)
    ks_stat = float(np.max(np.abs(pos_cdf - neg_cdf)))
    return round(ks_stat, 4)


def gini_from_auc(auc: float) -> float:
    return round(2 * auc - 1, 4)


def psi_alert(psi: float) -> str:
    if psi < 0.10:
        return "OK stable"
    elif psi < 0.25:
        return "WARN investigate"
    else:
        return "ALERT retrain"


# ── Synthetic data generator (replaces real model outputs in demo mode) ────────

def generate_synthetic_data(
    n_windows: int = 8,
    n_dev: int = 40_000,
    n_per_window: int = 5_000,
    seed: int = 42,
) -> tuple:
    """
    Generates realistic synthetic credit score data with gradual drift.

    Simulates:
      - Development period: scores from a well-calibrated model
      - Monitoring windows: increasing drift (concept + covariate shift)
        mimicking a model that was trained in 2019 and monitored into 2021.

    Returns:
      dev_scores  : (n_dev,)   - model scores on development data
      dev_labels  : (n_dev,)   - true default labels on development data
      windows     : list of dicts with keys 'label', 'scores', 'labels', 'features'
    """
    rng = np.random.default_rng(seed)

    # Development period: well-calibrated model
    # Scores are beta-distributed; defaults have lower scores
    def_rate_dev = 0.12
    n_def_dev = int(n_dev * def_rate_dev)

    # Non-defaults: higher scores (beta skewed right)
    non_def_scores = rng.beta(2.5, 4.0, size=n_dev - n_def_dev)  # scores cluster higher
    # Defaults: lower scores (beta skewed left)
    def_scores = rng.beta(3.5, 2.5, size=n_def_dev)              # scores cluster lower

    dev_scores = np.concatenate([non_def_scores, def_scores])
    dev_labels = np.concatenate([np.zeros(n_dev - n_def_dev), np.ones(n_def_dev)])

    # Shuffle
    idx = rng.permutation(len(dev_scores))
    dev_scores, dev_labels = dev_scores[idx], dev_labels[idx]

    # Simulated key feature: credit utilisation (pct) - normally ~45%, drifts upward
    dev_util = rng.beta(2, 2.5, size=n_dev) * 100  # 0-100, peak ~44%

    # Monitoring windows introduce gradual score compression and rising default rates
    start_date = date(2020, 1, 1)
    window_labels = pd.date_range(start=start_date, periods=n_windows, freq="MS").strftime("%Y-%m")
    windows = []

    for i, label in enumerate(window_labels):

        # Drift parameters - worsen gradually
        drift_factor = i / n_windows                         # 0 -> 1 over the windows
        default_rate_w = 0.12 + 0.08 * drift_factor         # rises from 12% -> 20%
        score_compression = 1 - 0.30 * drift_factor         # scores compress toward 0.5
        util_drift = 12 * drift_factor                       # utilisation rises

        n_def = int(n_per_window * default_rate_w)
        n_non = n_per_window - n_def

        non_def_w = rng.beta(2.5, 4.0, size=n_non) * score_compression + (1 - score_compression) * 0.5
        def_w = rng.beta(3.5, 2.5, size=n_def) * score_compression + (1 - score_compression) * 0.5

        w_scores = np.concatenate([non_def_w, def_w])
        w_labels = np.concatenate([np.zeros(n_non), np.ones(n_def)])
        w_idx    = rng.permutation(len(w_scores))
        w_scores, w_labels = w_scores[w_idx], w_labels[w_idx]

        # Feature: credit utilisation with drift
        w_util = rng.beta(2 + util_drift * 0.1, 2.5 - util_drift * 0.05, size=n_per_window) * 100

        windows.append({
            "label": label,
            "scores": w_scores,
            "labels": w_labels,
            "util": w_util,
        })

    return dev_scores, dev_labels, dev_util, windows


# ── Main monitoring loop ───────────────────────────────────────────────────────

def run_monitoring(
    dev_scores:  np.ndarray,
    dev_labels:  np.ndarray,
    dev_util:    np.ndarray,
    windows:     list,
) -> pd.DataFrame:

    dev_auc = float(roc_auc_score(dev_labels, dev_scores))
    dev_gini = gini_from_auc(dev_auc)
    dev_ks   = compute_ks(dev_labels, dev_scores)
    dev_ece  = compute_ece(dev_labels, dev_scores)

    print(f"\n{'═'*72}")
    print("  DEVELOPMENT PERIOD BASELINE")
    print(f"{'═'*72}")
    print(f"  N samples : {len(dev_scores):,}")
    print(f"  Default rate : {dev_labels.mean():.1%}")
    print(f"  AUC  : {dev_auc:.4f}   Gini : {dev_gini:.4f}   KS : {dev_ks:.4f}   ECE : {dev_ece:.4f}")
    print(f"  Score p10/p50/p90 : "
          f"{np.percentile(dev_scores,10):.3f} / "
          f"{np.percentile(dev_scores,50):.3f} / "
          f"{np.percentile(dev_scores,90):.3f}")

    records = []

    print(f"\n{'═'*72}")
    print("  MONITORING WINDOWS")
    print(f"{'─'*72}")
    print(f"  {'Window':<10} {'N':>6} {'DefRate':>8} {'ScorePSI':>10} {'UtilPSI':>9} "
          f"{'AUC':>7} {'Gini':>7} {'KS':>7} {'ECE':>7}  Alert")
    print(f"{'─'*72}")

    for w in windows:
        score_psi = compute_psi(dev_scores, w["scores"])
        util_psi  = compute_psi(dev_util,   w["util"])
        w_auc = float(roc_auc_score(w["labels"], w["scores"]))
        w_gini    = gini_from_auc(w_auc)
        w_ks      = compute_ks(w["labels"], w["scores"])
        w_ece     = compute_ece(w["labels"], w["scores"])
        def_rate  = w["labels"].mean()
        alert     = psi_alert(score_psi)

        print(f"  {w['label']:<10} {len(w['scores']):>6,} {def_rate:>8.1%} "
              f"{score_psi:>10.4f} {util_psi:>9.4f} "
              f"{w_auc:>7.4f} {w_gini:>7.4f} {w_ks:>7.4f} {w_ece:>7.4f}  {alert}")

        records.append({
            "window":           w["label"],
            "n_samples":        len(w["scores"]),
            "default_rate":     round(float(def_rate), 4),
            "score_psi":        score_psi,
            "util_psi":         util_psi,
            "auc":              round(float(w_auc), 4),
            "gini":             round(float(w_gini), 4),
            "ks_stat":          w_ks,
            "ece":              w_ece,
            "auc_delta_vs_dev": round(float(w_auc - dev_auc), 4),
            "score_p50":        round(float(np.percentile(w["scores"], 50)), 4),
            "score_p90":        round(float(np.percentile(w["scores"], 90)), 4),
            "psi_alert": psi_alert(score_psi),
        })

    df = pd.DataFrame(records)

    print(f"{'─'*72}")
    print(f"  Baseline AUC: {dev_auc:.4f}  |  Baseline Gini: {dev_gini:.4f}")
    print(f"  AUC decline triggers retrain recommendation if > 0.03 pp below baseline")

    alerts = df[df["psi_alert"].str.contains("WARN|ALERT", na=False)]
    if len(alerts):
        print(f"\n  {len(alerts)} window(s) exceed PSI thresholds:")
        for _, row in alerts.iterrows():
            print(f"     {row['window']}: score PSI={row['score_psi']:.4f} -> {row['psi_alert']}")
    else:
        print("\n  All windows within stable PSI thresholds")

    return df


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=8,
                        help="Number of monthly monitoring windows (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="monitoring_report.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print("\nCredit Risk Model Monitoring")
    print("Running with synthetic data - replace generate_synthetic_data()")
    print("with your own (dev_scores, dev_labels, dev_features, windows) arrays.")

    dev_scores, dev_labels, dev_util, windows = generate_synthetic_data(
        n_windows=args.windows, seed=args.seed
    )

    report_df = run_monitoring(dev_scores, dev_labels, dev_util, windows)

    report_df.to_csv(args.output, index=False)
    print(f"\n  Report saved -> {args.output}\n")

    return report_df


if __name__ == "__main__":
    main()
