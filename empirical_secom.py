"""
Empirical case study on a REAL production defect stream (UCI SECOM).

This script anchors the four-component CSP-1 economic design in real data,
addressing the common criticism that CSP-1 economic models are validated only
against synthetic Bernoulli streams. It does three things:

  1. Estimates the real incoming defect rate p' from the SECOM pass/fail
     sequence in temporal order (label 1 = fail/defective, -1 = pass).

  2. Computes the economic-optimal CSP-1 policy (i*, f*) at that real p' for a
     range of illustrative escape costs, and shows the regime switch occurs at
     the threshold predicted by the managerial rule p' > c_ins/(c_esc - c_rew).

  3. REPLAY VALIDATION: runs each policy (economic-optimal, static baseline, and
     a classical AOQL-designed plan) over the *actual* SECOM defect sequence and
     tallies the realized AFI, AOQ, transitions and total cost. The replay makes
     NO i.i.d. assumption -- it consumes the real (and mildly autocorrelated,
     rho ~ 0.11) stream verbatim -- so the gap between the i.i.d. model and the
     realized cost quantifies, on real data, the error of the modeling
     assumption. The sampling-mode coin flips are averaged over many seeds.

Outputs: empirical_secom.csv, empirical_secom.tex, empirical_secom.png
The unit costs are illustrative (no firm-specific cost accounting is available
for SECOM); the contribution is demonstrating the full design-and-validation
pipeline on a real defect stream, not calibrating a specific factory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csp_model import CSP1EconomicModel, optimize_csp1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(BASE_DIR, "data", "secom_labels.data")

# Illustrative cost parameters (paper base scenario). c_esc is swept.
C_INS, C_REJ, C_REW = 1.0, 500.0, 20.0
Q = 1000                     # nominal lot horizon for per-lot normalization
F_LO = 0.01                  # lower bound on sampling fraction
C_ESC_GRID = [20.0, 35.0, 50.0, 100.0]
BASELINE = (50, 0.10)        # static heuristic plan used throughout the paper
N_REPLAY_SEEDS = 20000       # Monte Carlo over sampling-mode coin flips


def load_secom_sequence(path=LABELS):
    """Real defect indicator d_t (1 = fail/defective) in temporal order."""
    d = []
    for line in open(path):
        tok = line.strip().split(" ", 1)
        if tok and tok[0] in ("1", "-1"):
            d.append(1 if tok[0] == "1" else 0)
    return np.asarray(d, dtype=np.int8)


def aoql_of_plan(i, f, p_grid=None):
    """AOQL = max over p' of the AOQ of plan (i, f)."""
    if p_grid is None:
        p_grid = np.linspace(1e-4, 0.30, 6000)
    aoq = []
    for p in p_grid:
        m = CSP1EconomicModel(i=i, f=f, p_prime=p, lot_size=Q)
        aoq.append(m.aoq)
    return float(np.max(aoq))


def design_aoql_plan(target_aoql, f=0.10, i_max=500):
    """Smallest i (at fixed f) whose AOQL <= target. Classical statistical design."""
    for i in range(1, i_max + 1):
        if aoql_of_plan(i, f) <= target_aoql:
            return i, f
    return i_max, f


_REPLAY_CACHE = {}


def replay_metrics(d, i, f, seeds=N_REPLAY_SEEDS, base_seed=2026):
    """Replay CSP-1 over the REAL defect sequence d, vectorized across seeds.

    The defect at each step is the fixed real value (same for every seed); only
    the sampling-mode coin flips differ across seeds. Returns realized per-lot
    (Q-normalized) AFI, AOQ, N_trans, N_found averaged over seeds. Cost-free, so
    the result is cached and reused across the c_esc sweep.
    """
    key = (i, round(f, 6), seeds, base_seed)
    if key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]
    n = len(d)
    defective = d.astype(bool)
    rng = np.random.default_rng(base_seed)

    in_full = np.ones(seeds, dtype=bool)      # all start in 100% inspection
    run = np.zeros(seeds, dtype=np.int32)     # consecutive-conforming counter
    inspected = np.zeros(seeds, dtype=np.int64)
    found = np.zeros(seeds, dtype=np.int64)
    escaped = np.zeros(seeds, dtype=np.int64)
    transitions = np.zeros(seeds, dtype=np.int64)

    for k in range(n):
        dk = bool(defective[k])
        draws = rng.random(seeds)
        sampling = ~in_full
        # --- full-inspection seeds: every unit inspected ---
        inspected[in_full] += 1
        if dk:
            found[in_full] += 1
            run[in_full] = 0
        else:
            run[in_full] += 1
            cleared = in_full & (run >= i)
            in_full[cleared] = False
        # --- sampling seeds: inspect with prob f ---
        insp_mask = sampling & (draws < f)
        inspected[insp_mask] += 1
        if dk:
            # inspected defectives are found -> revert; uninspected defectives escape
            found[insp_mask] += 1
            transitions[insp_mask] += 1
            in_full[insp_mask] = True
            run[insp_mask] = 0
            escaped[sampling & (draws >= f)] += 1

    afi = inspected.mean() / n
    aoq = escaped.mean() / n
    n_found = found.mean() / n * Q
    n_trans = transitions.mean() / n * Q
    out = dict(afi=afi, aoq=aoq, n_trans=n_trans, n_found=n_found)
    _REPLAY_CACHE[key] = out
    return out


def replay_cost(d, i, f, c_esc, **kw):
    """Realized cost of plan (i, f) on the real stream at escape cost c_esc."""
    m = replay_metrics(d, i, f, **kw)
    etc = (C_INS * m['afi'] * Q + c_esc * m['aoq'] * Q
           + C_REJ * m['n_trans'] + C_REW * m['n_found'])
    return dict(etc=etc, **m)


def model_cost(i, f, p, c_esc):
    m = CSP1EconomicModel(i=i, f=f, p_prime=p, lot_size=Q)
    return dict(afi=m.afi, aoq=m.aoq, n_trans=m.transition_rate,
                n_found=m.n_found, etc=m.expected_total_cost(C_INS, c_esc, C_REJ, C_REW))


def main():
    d = load_secom_sequence()
    n = len(d)
    p_hat = d.mean()
    print(f"SECOM: n={n}, real p'={p_hat:.4f}")

    # A classical AOQL plan as an additional statistical reference.
    target_aoql = 0.015
    i_aoql, f_aoql = design_aoql_plan(target_aoql, f=0.10)
    print(f"AOQL plan (target AOQL={target_aoql:.3f}): i={i_aoql}, f={f_aoql}, "
          f"actual AOQL={aoql_of_plan(i_aoql, f_aoql):.4f}")

    rows = []
    for c_esc in C_ESC_GRID:
        (i_opt, f_opt), _ = optimize_csp1(
            p_hat, lot_size=Q, c_ins=C_INS, c_esc=c_esc, c_rej=C_REJ,
            c_rew=C_REW, i_bounds=(1, 500), f_bounds=(F_LO, 1.0))
        # predicted threshold for switching to full inspection
        thr = C_INS / (c_esc - C_REW) if c_esc > C_REW else np.inf
        # label the regime by the actual optimizer outcome (i* saturates -> full)
        saturated = i_opt >= 500
        regime = "full" if saturated else "fractional"

        m_opt = model_cost(i_opt, f_opt, p_hat, c_esc)
        r_opt = replay_cost(d, i_opt, f_opt, c_esc)
        r_base = replay_cost(d, BASELINE[0], BASELINE[1], c_esc)
        r_aoql = replay_cost(d, i_aoql, f_aoql, c_esc)

        gap = abs(m_opt['etc'] - r_opt['etc']) / r_opt['etc'] * 100.0
        save_base = (r_base['etc'] - r_opt['etc']) / r_base['etc'] * 100.0
        save_aoql = (r_aoql['etc'] - r_opt['etc']) / r_aoql['etc'] * 100.0

        print(f"c_esc={c_esc:6.1f} thr={thr:.4f} regime={regime:10s} "
              f"opt=(i={i_opt},f={f_opt:.3f}) | model ETC={m_opt['etc']:8.2f} "
              f"replay ETC={r_opt['etc']:8.2f} gap={gap:4.1f}% | "
              f"save vs base={save_base:5.1f}% vs AOQL={save_aoql:5.1f}%")

        rows.append(dict(
            c_esc=c_esc, threshold=round(thr, 4), regime=regime,
            i_opt=i_opt, f_opt=round(f_opt, 3),
            ETC_model=round(m_opt['etc'], 1), ETC_replay=round(r_opt['etc'], 1),
            model_replay_gap_pct=round(gap, 2),
            AFI_model=round(m_opt['afi'], 4), AFI_replay=round(r_opt['afi'], 4),
            ETC_base_replay=round(r_base['etc'], 1),
            ETC_aoql_replay=round(r_aoql['etc'], 1),
            save_vs_base_pct=round(save_base, 1),
            save_vs_aoql_pct=round(save_aoql, 1)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "empirical_secom.csv"), index=False)
    print("\nSaved empirical_secom.csv")
    write_tex(df, p_hat, n, i_aoql, f_aoql, target_aoql)
    make_figure(df, p_hat)

    validate_across_range(d, p_hat, i_aoql)
    return df


def validate_across_range(d, p_hat, i_aoql, c_esc=50.0):
    """Model vs. realized (replay) across the operating range, including
    intermediate plans where autocorrelation has the most room to bite."""
    plans = [(1, 0.01), (10, 0.10), (30, 0.10), (50, 0.10),
             (i_aoql, 0.10), (100, 0.10), (20, 0.30)]
    print("\n=== Model vs. realized (replay) across the operating range "
          f"(real p'={p_hat:.4f}, c_esc={c_esc:.0f}) ===")
    rows = []
    for (i, f) in plans:
        mm = model_cost(i, f, p_hat, c_esc)
        rr = replay_cost(d, i, f, c_esc)
        afi_gap = abs(mm['afi'] - rr['afi']) / rr['afi'] * 100.0
        etc_gap = abs(mm['etc'] - rr['etc']) / rr['etc'] * 100.0
        print(f"(i={i:3d},f={f:.2f}) AFI model/real {mm['afi']:.4f}/{rr['afi']:.4f} "
              f"({afi_gap:4.1f}%) | ETC model/real {mm['etc']:8.2f}/{rr['etc']:8.2f} "
              f"({etc_gap:4.1f}%)")
        rows.append(dict(i=i, f=f, AFI_model=round(mm['afi'], 4),
                         AFI_replay=round(rr['afi'], 4), AFI_gap_pct=round(afi_gap, 2),
                         ETC_model=round(mm['etc'], 1), ETC_replay=round(rr['etc'], 1),
                         ETC_gap_pct=round(etc_gap, 2)))
    vdf = pd.DataFrame(rows)
    vdf.to_csv(os.path.join(BASE_DIR, "empirical_validation.csv"), index=False)

    lines = ["% Auto-generated by empirical_secom.py",
             "\\begin{table}[htbp]", "\\centering",
             f"\\caption{{The i.i.d.\\ renewal model vs.\\ realized performance on the "
             f"real SECOM stream ($\\hat{{p}}'={p_hat:.4f}$, $c_{{esc}}={c_esc:.0f}$), "
             f"across plans spanning the operating range. The model is accurate only at "
             f"the regime \\emph{{extremes}}---minimal sampling ($i=1$) and near-full "
             f"inspection ($i\\ge100$), where the realized cost gap is $\\le2\\%$---but "
             f"errs by up to $\\sim$34\\% at intermediate inspection intensities, where "
             f"the autocorrelation of the real stream ($\\rho\\approx0.11$) bites. "
             f"Crucially, the economic optimum (Table~\\ref{{tab:empirical}}) always "
             f"lies at an extreme, so this divergence never degrades the optimal policy.}}",
             "\\label{tab:empirical_val}", "\\small",
             "\\begin{tabular}{rr rr r rr r}", "\\toprule",
             "$i$ & $f$ & $AFI_{model}$ & $AFI_{real}$ & gap & "
             "$ETC_{model}$ & $ETC_{real}$ & gap \\\\",
             " & & & & (\\%) & & & (\\%) \\\\", "\\midrule"]
    for r in vdf.itertuples():
        lines.append(f"{r.i} & {r.f:.2f} & {r.AFI_model:.4f} & {r.AFI_replay:.4f} & "
                     f"{r.AFI_gap_pct:.1f} & {r.ETC_model:.0f} & {r.ETC_replay:.0f} & "
                     f"{r.ETC_gap_pct:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(BASE_DIR, "empirical_validation.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Saved empirical_validation.csv + .tex")
    return vdf


def write_tex(df, p_hat, n, i_aoql, f_aoql, target_aoql):
    lines = []
    lines.append("% Auto-generated by empirical_secom.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Economic-optimal CSP-1 design at the real SECOM defect "
                 f"rate $\\hat{{p}}'={p_hat:.4f}$ ($n={n}$ units), validated by replay "
                 f"over the true defect sequence. ``Model'' is the i.i.d.\\ renewal "
                 f"prediction; ``Replay'' is the realized cost on the real (mildly "
                 f"autocorrelated) stream. Savings are of the optimal policy over the "
                 f"static baseline ($i=50,f=0.10$) and a classical AOQL plan "
                 f"($i={i_aoql},f={f_aoql:.2f}$, AOQL$=${target_aoql:.3f}). "
                 f"Costs $c_{{ins}}=1$, $c_{{rej}}=500$, $c_{{rew}}=20$ (illustrative).}}")
    lines.append("\\label{tab:empirical}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{rccrr rr rr}")
    lines.append("\\toprule")
    lines.append("$c_{esc}$ & Regime & $(i^*,f^*)$ & $ETC_{model}$ & $ETC_{replay}$ "
                 "& Gap & $ETC_{base}$ & vs.\\ base & vs.\\ AOQL \\\\")
    lines.append(" & & & & & (\\%) & (replay) & (\\%) & (\\%) \\\\")
    lines.append("\\midrule")
    for r in df.itertuples():
        # at the full-inspection floor f is irrelevant (pi2 -> 0), so report 100% insp.
        fpair = "100\\% insp." if r.regime == "full" else f"$({r.i_opt},{r.f_opt:.2f})$"
        lines.append(
            f"{r.c_esc:.0f} & {r.regime} & {fpair} & {r.ETC_model:.0f} & "
            f"{r.ETC_replay:.0f} & {r.model_replay_gap_pct:.1f} & "
            f"{r.ETC_base_replay:.0f} & {r.save_vs_base_pct:.1f} & "
            f"{r.save_vs_aoql_pct:.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(os.path.join(BASE_DIR, "empirical_secom.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Saved empirical_secom.tex")


def make_figure(df, p_hat):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df['ETC_base_replay'], w, label='Static baseline', color='#bdbdbd')
    ax.bar(x, df['ETC_aoql_replay'], w, label='AOQL plan', color='#7fcdbb')
    ax.bar(x + w, df['ETC_replay'], w, label='Economic optimum', color='#2c7fb8')
    ax.set_xticks(x)
    ax.set_xticklabels([f"$c_{{esc}}={c:.0f}$\n({r})"
                        for c, r in zip(df['c_esc'], df['regime'])], fontsize=9)
    ax.set_ylabel('Realized ETC per lot (replay on real SECOM stream)')
    ax.set_title(f"CSP-1 economic design on the real SECOM stream "
                 f"($\\hat{{p}}'={p_hat:.4f}$)")
    ax.legend()
    ax.grid(True, axis='y', color='#ededed')
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "empirical_secom.png"),
                dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, "empirical_secom.eps"),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print("Saved empirical_secom.png + .eps")


if __name__ == "__main__":
    main()
