"""
TASK B1 - Robustness of the economic CSP-1 design to misestimation of p'.

A practitioner never knows the true defect rate p'; they act on an estimate
p_hat. We quantify the economic cost of that estimation error. For a perturbation
delta in {-30,...,+30}% we:
  1. Compute the optimal policy (i*, f*) ASSUMING the estimated p_hat = p'(1+delta).
  2. Evaluate the ETC actually INCURRED by running that policy under the TRUE p'.
  3. Compare against the perfect-information optimum (policy optimized for the
     true p'), reporting the % cost excess of acting on a wrong estimate.

Scenario design (decision documented for the paper):
  The optimal policy is a near-step function of p' (Proposition 1): below a
  defect-rate threshold the optimum is minimal inspection (i*=1, f*=0.01); above
  it the optimum jumps to the 100% floor. Consequently misestimation is harmless
  *within* a regime and only bites once the estimate crosses the phase boundary.
  We therefore report three scenarios sharing c_rej=500, c_ins=1, c_rew=20:
    S1 (p'=0.05, c_esc=50): firmly INSIDE the full-inspection regime -> robust.
    S2 (p'=0.06, c_esc=35): just past the boundary -> moderate, asymmetric penalty.
    S3 (p'=0.07, c_esc=35): further past it -> larger penalty for under-inspection.
  The originally proposed (p'=0.05, c_esc=50) is S1; on its own it yields a
  perfectly flat (and uninformative) curve, which is itself the "robust interior"
  message. S2/S3 expose the asymmetric danger of UNDER-estimating p'.

Outputs:
  - robustness_results.csv
  - robustness_pprime.png  (x = % error in p', y = % cost excess)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from csp_model import CSP1EconomicModel, optimize_csp1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

C_INS = 1.0
C_REW = 20.0
Q = 1000
PERTURBATIONS = [-30, -20, -10, 0, 10, 20, 30]  # percent error in estimated p'

SCENARIOS = [
    {'label': 'S1: $p\'=0.05$, $c_{esc}=50$ (interior)',  'p_true': 0.05, 'c_esc': 50.0, 'c_rej': 500.0},
    {'label': 'S2: $p\'=0.06$, $c_{esc}=35$ (boundary)',  'p_true': 0.06, 'c_esc': 35.0, 'c_rej': 500.0},
    {'label': 'S3: $p\'=0.07$, $c_{esc}=35$ (past edge)',  'p_true': 0.07, 'c_esc': 35.0, 'c_rej': 500.0},
]


def incurred_etc(policy_i, policy_f, p_true, c_esc, c_rej):
    """ETC actually incurred when running (policy_i, policy_f) under the true p'."""
    m = CSP1EconomicModel(i=policy_i, f=policy_f, p_prime=p_true, lot_size=Q)
    return m.expected_total_cost(c_ins=C_INS, c_esc=c_esc, c_rej=c_rej, c_rew=C_REW)


def run_scenario(p_true, c_esc, c_rej):
    (i_perf, f_perf), _ = optimize_csp1(
        p_true, lot_size=Q, c_ins=C_INS, c_esc=c_esc, c_rej=c_rej, c_rew=C_REW)
    etc_perfect = incurred_etc(i_perf, f_perf, p_true, c_esc, c_rej)

    rows = []
    for delta in PERTURBATIONS:
        p_est = p_true * (1 + delta / 100.0)
        (i_est, f_est), _ = optimize_csp1(
            p_est, lot_size=Q, c_ins=C_INS, c_esc=c_esc, c_rej=c_rej, c_rew=C_REW)
        etc_inc = incurred_etc(i_est, f_est, p_true, c_esc, c_rej)
        excess = (etc_inc / etc_perfect - 1.0) * 100.0
        rows.append({
            'p_true': p_true, 'c_esc': c_esc, 'c_rej': c_rej,
            'error_pct': delta, 'p_est': round(p_est, 4),
            'i_opt_est': i_est, 'f_opt_est': round(f_est, 4),
            'etc_incurred': round(etc_inc, 2), 'etc_perfect': round(etc_perfect, 2),
            'cost_excess_pct': round(excess, 3),
        })
    return (i_perf, f_perf, etc_perfect), rows


def run_robustness():
    all_rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['s--', 'o-', '^-']
    colors = ['#2c7fb8', '#c0392b', '#d35400']

    for sc, mk, col in zip(SCENARIOS, markers, colors):
        (i_perf, f_perf, etc_perf), rows = run_scenario(sc['p_true'], sc['c_esc'], sc['c_rej'])
        all_rows.extend(rows)

        print("=" * 88)
        print(f"{sc['label']}  ->  perfect optimum i*={i_perf}, f*={f_perf:.4f}, ETC*={etc_perf:.2f}")
        print(f"{'err(%)':>7} {'p_est':>8} {'i*_est':>7} {'f*_est':>8} "
              f"{'ETC_incurred':>13} {'excess(%)':>10}")
        for r in rows:
            print(f"{r['error_pct']:7d} {r['p_est']:8.4f} {r['i_opt_est']:7d} "
                  f"{r['f_opt_est']:8.4f} {r['etc_incurred']:13.2f} {r['cost_excess_pct']:10.3f}")

        xs = [r['error_pct'] for r in rows]
        ys = [r['cost_excess_pct'] for r in rows]
        ax.plot(xs, ys, mk, color=col, linewidth=2, markersize=7, label=sc['label'])

    print("=" * 88)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(BASE_DIR, 'robustness_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel("Estimation error in $\\hat{p}'$ (%)")
    ax.set_ylabel("Cost excess vs. perfect-information optimum (%)")
    ax.set_title("Robustness of the economic design to $p'$ misestimation")
    ax.legend(fontsize=9)
    # solid light gray instead of alpha (PostScript/EPS has no transparency)
    ax.grid(True, color='#e8e8e8')
    fig_path = os.path.join(BASE_DIR, 'robustness_pprime.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(os.path.join(BASE_DIR, 'robustness_pprime.eps'),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {fig_path} (300 dpi, RGB) + EPS")
    return df


if __name__ == "__main__":
    run_robustness()
