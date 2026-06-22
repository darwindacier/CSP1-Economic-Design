"""
FASE 1.2 - Discrete-event (unit-by-unit) simulation of the real CSP-1 process,
used to validate the corrected renewal-reward model in csp_model.py.

The simulator reproduces CSP-1 exactly:
  * Start in 100% inspection. Inspect every unit; count consecutive conforming
    units. A defective resets the count. On reaching i consecutive conforming,
    switch to fractional sampling.
  * In sampling mode, inspect each unit independently with probability f. If an
    inspected unit is defective, revert to 100% inspection (count reset to 0).
    Uninspected defectives escape.

For ~6 representative scenarios we compare AFI, AOQ, N_trans and ETC under
  (a) the OLD f-independent model (pi = p'/(p'+q^i)),
  (b) the NEW renewal-reward model (csp_model.CSP1EconomicModel),
  (c) the simulation,
confirming that (b) matches (c) within ~1-2% while (a) does not.

Outputs: model_validation.csv, model_validation.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csp_model import CSP1EconomicModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

C_INS, C_ESC, C_REJ, C_REW = 1.0, 50.0, 500.0, 20.0
Q = 1000

# Representative (i, f, p') triples that keep the sampling phase active so the
# f-dependence of the steady state is genuinely exercised.
SCENARIOS = [
    (5,  0.10, 0.02),
    (10, 0.05, 0.02),
    (20, 0.10, 0.05),
    (50, 0.10, 0.05),
    (15, 0.20, 0.01),
    (8,  0.50, 0.10),
]


def etc_from_metrics(afi, aoq, n_trans, n_found):
    return (C_INS * afi * Q + C_ESC * aoq * Q + C_REJ * n_trans + C_REW * n_found)


def old_model(i, f, p):
    """Original f-independent DTMC-balance model."""
    q = 1.0 - p
    qi = q ** i
    denom = p + qi
    pi1 = p / denom
    pi2 = qi / denom
    afi = pi1 + pi2 * f
    aoq = p * (1.0 - afi)
    n_found = p * afi * Q
    n_trans = pi2 * f * p * Q
    return afi, aoq, n_trans, n_found, etc_from_metrics(afi, aoq, n_trans, n_found)


def new_model(i, f, p):
    m = CSP1EconomicModel(i=i, f=f, p_prime=p, lot_size=Q)
    etc = m.expected_total_cost(C_INS, C_ESC, C_REJ, C_REW)
    return m.afi, m.aoq, m.transition_rate, m.n_found, etc


def simulate(i, f, p, n_units=2_000_000, seed=12345):
    """Unit-by-unit discrete-event simulation. Returns per-Q metrics."""
    rng = np.random.default_rng(seed)
    defective = rng.random(n_units) < p          # true defect indicator
    sample_draw = rng.random(n_units)             # used only in sampling mode

    inspected = 0
    found = 0
    escaped = 0
    transitions = 0

    in_full = True          # start in 100% inspection
    run = 0                 # consecutive conforming counter (full mode)

    for k in range(n_units):
        d = defective[k]
        if in_full:
            inspected += 1
            if d:
                found += 1
                run = 0
            else:
                run += 1
                if run >= i:
                    in_full = False     # clear -> switch to sampling
        else:
            if sample_draw[k] < f:      # this unit is inspected
                inspected += 1
                if d:
                    found += 1
                    transitions += 1
                    in_full = True      # revert to 100% inspection
                    run = 0
            else:                       # not inspected
                if d:
                    escaped += 1

    afi = inspected / n_units
    aoq = escaped / n_units
    n_trans = transitions / n_units * Q
    n_found = found / n_units * Q
    return afi, aoq, n_trans, n_found, etc_from_metrics(afi, aoq, n_trans, n_found)


def run_validation():
    rows = []
    print("=" * 110)
    print("MODEL VALIDATION: old (f-independent) vs new (renewal) vs simulation")
    print("=" * 110)
    for (i, f, p) in SCENARIOS:
        o = old_model(i, f, p)
        n = new_model(i, f, p)
        s = simulate(i, f, p)
        err_new = abs(n[4] - s[4]) / s[4] * 100.0
        err_old = abs(o[4] - s[4]) / s[4] * 100.0
        rows.append({
            'i': i, 'f': f, 'p': p,
            'AFI_old': round(o[0], 5), 'AFI_new': round(n[0], 5), 'AFI_sim': round(s[0], 5),
            'AOQ_old': round(o[1], 6), 'AOQ_new': round(n[1], 6), 'AOQ_sim': round(s[1], 6),
            'Ntrans_old': round(o[2], 4), 'Ntrans_new': round(n[2], 4), 'Ntrans_sim': round(s[2], 4),
            'ETC_old': round(o[4], 2), 'ETC_new': round(n[4], 2), 'ETC_sim': round(s[4], 2),
            'ETCerr_old_%': round(err_old, 2), 'ETCerr_new_%': round(err_new, 2),
        })
        print(f"i={i:3d} f={f:.2f} p={p:.2f} | AFI old/new/sim "
              f"{o[0]:.4f}/{n[0]:.4f}/{s[0]:.4f} | ETC old/new/sim "
              f"{o[4]:8.2f}/{n[4]:8.2f}/{s[4]:8.2f} | err old={err_old:5.2f}% new={err_new:4.2f}%")
    print("=" * 110)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(BASE_DIR, 'model_validation.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Max NEW-model ETC error vs simulation: {df['ETCerr_new_%'].max():.2f}%")
    print(f"Max OLD-model ETC error vs simulation: {df['ETCerr_old_%'].max():.2f}%")

    make_figure(df)
    return df


def make_figure(df):
    labels = [f"i={r.i},f={r.f}\np'={r.p}" for r in df.itertuples()]
    x = np.arange(len(df))
    w = 0.27
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # AFI panel
    ax1.bar(x - w, df['AFI_old'], w, label='Old model', color='#bdbdbd')
    ax1.bar(x,     df['AFI_new'], w, label='New model', color='#2c7fb8')
    ax1.bar(x + w, df['AFI_sim'], w, label='Simulation', color='#d95f02')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('AFI'); ax1.set_title('Average Fraction Inspected')
    ax1.legend(); ax1.grid(True, axis='y', color='#e8e8e8')

    # ETC panel
    ax2.bar(x - w, df['ETC_old'], w, label='Old model', color='#bdbdbd')
    ax2.bar(x,     df['ETC_new'], w, label='New model', color='#2c7fb8')
    ax2.bar(x + w, df['ETC_sim'], w, label='Simulation', color='#d95f02')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('ETC per lot'); ax2.set_title('Expected Total Cost')
    ax2.legend(); ax2.grid(True, axis='y', color='#e8e8e8')

    fig.suptitle('Model validation: renewal-reward model matches the simulated CSP-1 process')
    fig.tight_layout()
    fig_path = os.path.join(BASE_DIR, 'model_validation.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(os.path.join(BASE_DIR, 'model_validation.eps'),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {fig_path} (300 dpi, RGB) + EPS")


if __name__ == "__main__":
    run_validation()
