"""
experiments/ab_test_credit_policy.py
──────────────────────────────────────
Simulates a credit card acquisition policy experiment.

Business question
─────────────────
Should we approve credit card applications from the "near-prime" segment
(FICO 620–640) who we currently decline?

Experiment design
─────────────────
  Control   : current policy — decline all near-prime applicants
  Treatment : new policy     — approve near-prime applicants at 28% APR

Metrics
  Primary   : 12-month charge-off rate  (must not be unacceptably high)
  Secondary : net P&L per approved account

Statistical framework
  H0 : treatment charge-off rate ≤ control + tolerance (Δ)
  H1 : treatment charge-off rate > control + tolerance
  Two-proportion z-test, one-sided, α = 0.05

Usage:
  python experiments/ab_test_credit_policy.py
  python experiments/ab_test_credit_policy.py --n-per-arm 3000 --seed 7
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ── Experiment parameters ──────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # Policy parameters
    control_default_rate:    float = 0.08    # existing prime portfolio default rate (counterfactual)
    treatment_default_rate:  float = 0.16    # expected near-prime default rate at 28% APR
    max_tolerable_default:   float = 0.20    # reject policy if default rate > this

    # Financial parameters (per activated account, per year)
    apr:                     float = 0.28    # 28% APR on near-prime card
    avg_balance:             float = 900.0   # average revolving balance ($)
    lgd:                     float = 0.85    # loss given default (85% of balance)
    activation_rate:         float = 0.60   # share of approved applicants who activate card
    interchange_fee_annual:  float = 40.0   # annual interchange revenue per active account

    # Statistical parameters
    alpha:                   float = 0.05    # significance level
    power:                   float = 0.80    # desired test power (1 - β)
    n_per_arm:               int   = 2000    # override: if 0, use calculated n

    seed:                    int   = 42


# ── Statistical functions ──────────────────────────────────────────────────────

def power_calculation(p1: float, p2: float, alpha: float, power: float) -> int:
    """
    Sample size per arm for two-proportion z-test (one-sided).
    Returns n per arm to detect a difference |p1 - p2| with given power.
    """
    z_alpha = stats.norm.ppf(1 - alpha)          # critical value for α
    z_beta  = stats.norm.ppf(power)              # critical value for power (1-β)

    p_bar = (p1 + p2) / 2
    numerator   = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                   + z_beta  * math.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
    denominator = (p1 - p2) ** 2
    return math.ceil(numerator / denominator)


def two_prop_z_test(n1: int, x1: int, n2: int, x2: int) -> tuple[float, float]:
    """
    Two-proportion one-sided z-test.
    H0: p_treatment ≤ p_control
    H1: p_treatment > p_control
    Returns (z_stat, p_value).
    """
    p1 = x1 / n1   # control default rate
    p2 = x2 / n2   # treatment default rate
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p_value = 1 - stats.norm.cdf(z)   # one-sided, upper tail
    return round(z, 4), round(p_value, 5)


# ── P&L calculation ────────────────────────────────────────────────────────────

def compute_pnl(
    n_approved:      int,
    default_rate:    float,
    cfg:             ExperimentConfig,
) -> dict:
    """
    Net P&L per 1,000 approved applicants over 12 months.
    Revenue = activated accounts × (apr × avg_balance + interchange)
    Loss    = activated accounts × default_rate × avg_balance × lgd
    """
    n_activated    = n_approved * cfg.activation_rate
    n_defaults     = n_activated * default_rate
    n_performing   = n_activated - n_defaults

    interest_rev   = n_performing * cfg.apr * cfg.avg_balance
    interchange_rev= n_performing * cfg.interchange_fee_annual
    loss           = n_defaults   * cfg.avg_balance * cfg.lgd

    net_pnl        = interest_rev + interchange_rev - loss
    net_per_account= net_pnl / n_approved if n_approved > 0 else 0

    return {
        "n_approved":       n_approved,
        "n_activated":      round(n_activated),
        "n_defaults":       round(n_defaults),
        "interest_rev":     round(interest_rev, 2),
        "interchange_rev":  round(interchange_rev, 2),
        "loss":             round(loss, 2),
        "net_pnl":          round(net_pnl, 2),
        "net_per_account":  round(net_per_account, 2),
    }


# ── Sensitivity analysis ───────────────────────────────────────────────────────

def sensitivity_analysis(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Computes net P&L per 1,000 approved accounts across a range of default rates.
    Identifies the break-even default rate.
    """
    rows = []
    for dr in np.arange(0.08, 0.35, 0.02):
        pnl = compute_pnl(1000, dr, cfg)
        rows.append({
            "default_rate_pct": round(dr * 100, 1),
            "net_pnl_per_1k":   pnl["net_pnl"],
            "net_per_account":  pnl["net_per_account"],
            "profitable":       "✅" if pnl["net_pnl"] > 0 else "❌",
        })
    return pd.DataFrame(rows)


# ── Simulate experiment data ───────────────────────────────────────────────────

def simulate_experiment(cfg: ExperimentConfig, n_per_arm: int) -> dict:
    rng = np.random.default_rng(cfg.seed)

    # Control: counterfactual (what would have happened if we'd approved them)
    control_defaults = rng.binomial(1, cfg.control_default_rate, size=n_per_arm)

    # Treatment: near-prime applicants at expanded policy
    treat_defaults = rng.binomial(1, cfg.treatment_default_rate, size=n_per_arm)

    return {
        "n_control":  n_per_arm,
        "n_treatment":n_per_arm,
        "x_control":  int(control_defaults.sum()),
        "x_treatment":int(treat_defaults.sum()),
        "obs_control_rate": round(control_defaults.mean(), 4),
        "obs_treat_rate":   round(treat_defaults.mean(),   4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cfg: ExperimentConfig):
    W = 72
    print(f"\n{'═'*W}")
    print("  CREDIT CARD ACQUISITION POLICY EXPERIMENT")
    print(f"{'═'*W}")

    # ── Step 1: Power calculation ──────────────────────────────────────────────
    n_required = power_calculation(
        cfg.control_default_rate,
        cfg.treatment_default_rate,
        cfg.alpha,
        cfg.power,
    )
    print(f"\n{'─'*W}")
    print("  STEP 1 — Power Calculation")
    print(f"{'─'*W}")
    print(f"  Hypothesis  : H0: treatment default rate ≤ {cfg.max_tolerable_default:.0%}")
    print(f"  Control baseline default rate  : {cfg.control_default_rate:.1%}")
    print(f"  Expected treatment default rate: {cfg.treatment_default_rate:.1%}")
    print(f"  MDE (minimum detectable effect): {cfg.treatment_default_rate - cfg.control_default_rate:.1%}")
    print(f"  α = {cfg.alpha}   Power (1−β) = {cfg.power:.0%}")
    print(f"\n  Required sample size : {n_required:,} per arm  ({n_required*2:,} total)")

    n_per_arm = cfg.n_per_arm if cfg.n_per_arm > 0 else n_required
    print(f"  Using               : {n_per_arm:,} per arm  ({n_per_arm*2:,} total)")
    if n_per_arm < n_required:
        print(f"  ⚠️  Under-powered: {n_per_arm} < {n_required} required. "
              f"Estimated power: {n_per_arm/n_required*cfg.power:.0%}")

    # ── Step 2: Simulate experiment ────────────────────────────────────────────
    results = simulate_experiment(cfg, n_per_arm)

    print(f"\n{'─'*W}")
    print("  STEP 2 — Experiment Results")
    print(f"{'─'*W}")
    print(f"  {'':30} {'Control':>12} {'Treatment':>12}")
    print(f"  {'─'*54}")
    print(f"  {'N applicants':30} {results['n_control']:>12,} {results['n_treatment']:>12,}")
    print(f"  {'Defaults':30} {results['x_control']:>12,} {results['x_treatment']:>12,}")
    print(f"  {'Observed default rate':30} {results['obs_control_rate']:>12.1%} {results['obs_treat_rate']:>12.1%}")

    # ── Step 3: Statistical test ───────────────────────────────────────────────
    z_stat, p_value = two_prop_z_test(
        results["n_control"],  results["x_control"],
        results["n_treatment"],results["x_treatment"],
    )
    significant = p_value < cfg.alpha
    exceed_tolerance = results["obs_treat_rate"] > cfg.max_tolerable_default

    print(f"\n{'─'*W}")
    print("  STEP 3 — Statistical Test  (one-sided z-test, H1: treatment > control)")
    print(f"{'─'*W}")
    print(f"  z-statistic : {z_stat:>8.4f}")
    print(f"  p-value     : {p_value:>8.5f}  (threshold: {cfg.alpha})")
    print(f"  Significant : {'YES ❌' if significant else 'NO  ✅'}")
    print(f"  95% CI on treatment rate: "
          f"[{results['obs_treat_rate'] - 1.96*math.sqrt(results['obs_treat_rate']*(1-results['obs_treat_rate'])/n_per_arm):.3f}, "
          f"{results['obs_treat_rate'] + 1.96*math.sqrt(results['obs_treat_rate']*(1-results['obs_treat_rate'])/n_per_arm):.3f}]")
    print(f"  Exceeds tolerance ({cfg.max_tolerable_default:.0%}): {'YES ❌' if exceed_tolerance else 'NO  ✅'}")

    # ── Step 4: P&L analysis ───────────────────────────────────────────────────
    pnl_treat = compute_pnl(1000, results["obs_treat_rate"], cfg)

    print(f"\n{'─'*W}")
    print("  STEP 4 — P&L Analysis  (per 1,000 approved applicants, 12 months)")
    print(f"{'─'*W}")
    print(f"  Activation rate assumed     : {cfg.activation_rate:.0%}")
    print(f"  APR                         : {cfg.apr:.0%}")
    print(f"  Avg revolving balance       : ${cfg.avg_balance:,.0f}")
    print(f"  Loss given default (LGD)    : {cfg.lgd:.0%}")
    print(f"\n  {'Activated accounts':35} : {pnl_treat['n_activated']:>8,}")
    print(f"  {'Expected defaults':35} : {pnl_treat['n_defaults']:>8,}")
    print(f"  {'Interest revenue':35} : ${pnl_treat['interest_rev']:>12,.2f}")
    print(f"  {'Interchange revenue':35} : ${pnl_treat['interchange_rev']:>12,.2f}")
    print(f"  {'Expected losses':35} : ${pnl_treat['loss']:>12,.2f}")
    print(f"  {'─'*50}")
    print(f"  {'NET P&L':35} : ${pnl_treat['net_pnl']:>12,.2f}  "
          f"({'PROFITABLE ✅' if pnl_treat['net_pnl'] > 0 else 'LOSS ❌'})")
    print(f"  {'Net P&L per approved account':35} : ${pnl_treat['net_per_account']:>8.2f}")

    # ── Step 5: Sensitivity analysis ──────────────────────────────────────────
    sensitivity = sensitivity_analysis(cfg)
    breakeven_rows = sensitivity[sensitivity["net_pnl_per_1k"] >= 0]
    breakeven_rows = sensitivity[sensitivity["net_pnl_per_1k"] >= 0]
    breakeven_dr = float(breakeven_rows["default_rate_pct"].iloc[-1]) if len(breakeven_rows) > 0 else 0.0

    print(f"\n{'─'*W}")
    print("  STEP 5 — Sensitivity Analysis  (net P&L per 1,000 approved accounts)")
    print(f"{'─'*W}")
    print(sensitivity.to_string(index=False))
    print(f"\n  Break-even default rate: ~{breakeven_dr:.1f}%  "
          f"(policy is unprofitable above this threshold)")

    # ── Step 6: Decision recommendation ───────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  STEP 6 — Decision Recommendation")
    print(f"{'─'*W}")

    proceed = (not exceed_tolerance) and pnl_treat["net_pnl"] > 0

    if proceed:
        print(f"  ✅ PROCEED with expanded policy")
        print(f"     Default rate ({results['obs_treat_rate']:.1%}) is within tolerance ({cfg.max_tolerable_default:.0%})")
        print(f"     Positive net P&L of ${pnl_treat['net_pnl']:,.0f} per 1,000 approved accounts")
    else:
        reasons = []
        if exceed_tolerance:
            reasons.append(f"default rate {results['obs_treat_rate']:.1%} exceeds tolerance {cfg.max_tolerable_default:.0%}")
        if pnl_treat["net_pnl"] <= 0:
            reasons.append(f"negative net P&L (${pnl_treat['net_pnl']:,.0f})")
        print(f"  ❌ DO NOT PROCEED: {'; '.join(reasons)}")
        print(f"     Consider repricing (higher APR) or tightening the FICO threshold")

    print(f"\n{'═'*W}\n")

    # Return summary dict for programmatic use
    return {
        "n_required":         n_required,
        "n_used":             n_per_arm,
        "control_default":    results["obs_control_rate"],
        "treatment_default":  results["obs_treat_rate"],
        "z_stat":             z_stat,
        "p_value":            p_value,
        "significant":        significant,
        "net_pnl_per_1k":     pnl_treat["net_pnl"],
        "net_per_account":    pnl_treat["net_per_account"],
        "proceed":            proceed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-arm", type=int, default=2000,
                        help="Applicants per arm (0 = use calculated minimum)")
    parser.add_argument("--treatment-default-rate", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        n_per_arm=args.n_per_arm,
        treatment_default_rate=args.treatment_default_rate,
        seed=args.seed,
    )
    main(cfg)
