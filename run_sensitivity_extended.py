"""
TASK B2 - Sensitivity of the economic CSP-1 design to lot size Q and rework cost c_rew.

For the base cost scenario (c_esc=50, c_rej=500, c_ins=1) we sweep
  Q     in {100, 1000, 10000}
  c_rew in {10, 20, 40}
over the four incoming defect rates p in {0.01, 0.02, 0.05, 0.10}.

Two questions:
  (i)  Does the threshold structure survive (i* small at low p.c_esc, jumping to
       the 100% regime as p grows)?  -> reported via the i* column.
  (ii) How does the full-inspection cost floor scale?  Proposition 1 gives
       ETC_inf = c_ins.Q + c_rew.p'.Q = Q.(c_ins + c_rew.p'), i.e. it scales
       LINEARLY in Q and linearly in c_rew (weighted by p'). We verify the
       optimizer reaches this floor whenever the policy saturates.

Outputs:
  - sensitivity_extended.csv
  - sensitivity_extended.tex   (LaTeX table for the paper)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csp_model import CSP1EconomicModel, optimize_csp1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

C_INS = 1.0
C_ESC = 50.0
C_REJ = 500.0
P_VALUES = [0.01, 0.02, 0.05, 0.10]
Q_VALUES = [100, 1000, 10000]
CREW_VALUES = [10, 20, 40]


def asymptotic_floor(Q, c_rew, p):
    """Proposition 1 full-inspection floor: ETC_inf = Q (c_ins + c_rew p')."""
    return Q * (C_INS + c_rew * p)


def run_extended():
    rows = []
    print("=" * 96)
    print("EXTENDED SENSITIVITY: lot size Q and rework cost c_rew "
          f"(c_esc={C_ESC:.0f}, c_rej={C_REJ:.0f})")
    print("=" * 96)
    print(f"{'Q':>7} {'c_rew':>6} {'p':>6} {'i*':>5} {'f*':>8} "
          f"{'ETC*':>12} {'floor':>12} {'at floor?':>9}")
    print("-" * 96)

    for Q in Q_VALUES:
        for c_rew in CREW_VALUES:
            for p in P_VALUES:
                (i_opt, f_opt), cost = optimize_csp1(
                    p, lot_size=Q, c_ins=C_INS, c_esc=C_ESC, c_rej=C_REJ, c_rew=c_rew)
                floor = asymptotic_floor(Q, c_rew, p)
                at_floor = abs(cost - floor) / floor < 1e-3
                rows.append({
                    'Q': Q, 'c_rew': c_rew, 'p': p,
                    'i_opt': i_opt, 'f_opt': round(f_opt, 4),
                    'etc_opt': round(cost, 2), 'etc_floor': round(floor, 2),
                    'at_floor': at_floor,
                })
                print(f"{Q:7d} {c_rew:6d} {p:6.2f} {i_opt:5d} {f_opt:8.4f} "
                      f"{cost:12.2f} {floor:12.2f} {str(at_floor):>9}")
    print("=" * 96)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(BASE_DIR, 'sensitivity_extended.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    write_latex(df)
    make_figure()
    return df


def make_figure():
    """Support figure: how the rework cost c_rew shifts the inspection threshold.
    Fine p' grid at Q=1000 shows i*(p') jumping to the 100% regime, with the jump
    occurring at higher p' as c_rew increases."""
    Q = 1000
    p_grid = np.linspace(0.005, 0.12, 70)
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {10: ('o-', '#2c7fb8'), 20: ('s-', '#d95f02'), 40: ('^-', '#7570b3')}
    for c_rew in CREW_VALUES:
        i_stars = []
        for p in p_grid:
            (i_opt, _), _ = optimize_csp1(
                p, lot_size=Q, c_ins=C_INS, c_esc=C_ESC, c_rej=C_REJ, c_rew=c_rew)
            i_stars.append(i_opt)
        mk, col = styles[c_rew]
        ax.plot(p_grid, i_stars, mk, color=col, markersize=3, linewidth=1.5,
                label=f"$c_{{rew}}={c_rew}$")
    ax.set_xlabel("Incoming defect rate $p'$")
    ax.set_ylabel("Optimal clearance number $i^*$")
    ax.set_title(f"Threshold structure and its shift with rework cost\n"
                 f"($Q={Q}$, $c_{{esc}}={C_ESC:.0f}$, $c_{{rej}}={C_REJ:.0f}$)")
    ax.legend()
    # solid light gray instead of alpha (PostScript/EPS has no transparency)
    ax.grid(True, color='#e8e8e8')
    fig_path = os.path.join(BASE_DIR, 'sensitivity_extended.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(os.path.join(BASE_DIR, 'sensitivity_extended.eps'),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {fig_path} (300 dpi, RGB) + EPS")


def write_latex(df):
    """Compact grouped LaTeX table: one block per (Q, c_rew), four p-rows each."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Sensitivity of the optimal economic design to lot size $Q$ and "
                 "rework cost $c_{rew}$ (base scenario $c_{esc}=50$, $c_{rej}=500$, "
                 "$c_{ins}=1$). The threshold structure is preserved across all $(Q,c_{rew})$: "
                 "$i^*$ stays minimal for low $p'$ and saturates to the full-inspection regime "
                 "as $p'$ grows. When saturated, $ETC^*$ coincides with the Proposition~1 floor "
                 "$ETC_\\infty = Q\\,(c_{ins}+c_{rew}\\,p')$.}")
    lines.append("\\label{tab:sensitivity_extended}")
    lines.append("\\begin{tabular}{rrrrrrr}")
    lines.append("\\toprule")
    lines.append("$Q$ & $c_{rew}$ & $p'$ & $i^*$ & $f^*$ & $ETC^*$ & $ETC_\\infty$ \\\\")
    lines.append("\\midrule")

    prev_block = None
    for _, r in df.iterrows():
        block = (r['Q'], r['c_rew'])
        if prev_block is not None and block != prev_block:
            lines.append("\\midrule")
        q_str = f"{int(r['Q'])}" if block != prev_block else ""
        crew_str = f"{int(r['c_rew'])}" if block != prev_block else ""
        prev_block = block
        lines.append(
            f"{q_str} & {crew_str} & {r['p']:.2f} & {int(r['i_opt'])} & "
            f"{r['f_opt']:.4f} & {r['etc_opt']:.2f} & {r['etc_floor']:.2f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = os.path.join(BASE_DIR, 'sensitivity_extended.tex')
    with open(tex_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    run_extended()
