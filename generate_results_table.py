import os
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
from csp_model import CSP1EconomicModel


def compute_aoql(i, f, p_grid=2000):
    """Compute the AOQL (maximum AOQ) for a given CSP-1 plan (i, f).

    AOQL = max_{p in (0,1)} p * (1 - AFI(p; i, f))
    where AFI(p) = f / (f + (1-f)*(1-p)**i).
    We solve this numerically because the maximum is interior.
    """
    def aoq(p):
        if p <= 0 or p >= 1:
            return 0.0
        q_pow = (1.0 - p) ** i
        afi = f / (f + (1.0 - f) * q_pow)
        return p * (1.0 - afi)

    # Use bounded scalar maximization by minimizing negative AOQ
    def neg_aoq(p):
        return -aoq(p)

    res = minimize_scalar(neg_aoq, bounds=(1e-6, 1.0 - 1e-6), method='bounded')
    return float(-res.fun) if res.success else float(np.nan)


def format_row(row, cols):
    vals = []
    for c in cols:
        v = row[c]
        if c == 'p':
            vals.append(f"{float(v):.4f}")
        elif c == 'f_opt':
            v = float(v)
            if abs(v - 1.0) < 1e-4:
                vals.append("1.0")
            elif abs(v - 0.01) < 1e-4:
                vals.append("0.01")
            else:
                vals.append(f"{v:.4f}")
        elif c in ('aoq_opt', 'aoql'):
            vals.append(f"{float(v):.5f}")
        elif c in ('cost_opt', 'cost_static', 'reduction_%'):
            vals.append(f"{float(v):.2f}")
        else:
            vals.append(str(int(v)))
    return " & ".join(vals) + " \\\\"


def make_latex_table(df, cols, headers, col_format, caption, label, scale=None):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    if scale is not None:
        lines.append(f"\\resizebox{{{scale}\\textwidth}}{{!}}{{%")
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(format_row(row, cols))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if scale is not None:
        lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, 'sensitivity_results.csv')
    df = pd.read_csv(csv_path)

    # AOQ attained at (i*, f*) under the true p' of each scenario
    df['aoq_opt'] = [
        CSP1EconomicModel(i=int(r.i_opt), f=float(r.f_opt), p_prime=float(r.p)).aoq
        for r in df.itertuples()
    ]

    # AOQL of each optimal plan: max over all possible p of AOQ(p; i*,f*)
    df['aoql'] = [
        compute_aoql(i=int(r.i_opt), f=float(r.f_opt))
        for r in df.itertuples()
    ]

    # Representative subset for body: c_rej=500
    df_rep = df[df['c_rej'] == 500].copy()

    latex_rep = make_latex_table(
        df_rep,
        cols=['p', 'c_esc', 'i_opt', 'f_opt', 'cost_opt', 'cost_static', 'reduction_%'],
        headers=['$p\'$', '$c_{esc}$', '$i^*$', '$f^*$', '$ETC^*$', '$ETC_{base}$', 'Reduction (\\%)'],
        col_format='rrrrrrr',
        caption="Optimal CSP-1 parameters and cost reduction for $c_{rej}=500$ (representative subset). Full results, including the optimal-plan AOQL, in the Appendix (Table~\\ref{app:fulltable}).",
        label="tab:sensitivity"
    )

    latex_full = make_latex_table(
        df,
        cols=['p', 'c_esc', 'c_rej', 'i_opt', 'f_opt', 'aoq_opt', 'aoql', 'cost_opt', 'cost_static', 'reduction_%'],
        headers=['$p\'$', '$c_{esc}$', '$c_{rej}$', '$i^*$', '$f^*$', '$AOQ^*$', '$AOQL^*$', '$ETC^*$', '$ETC_{base}$', 'Reduction (\\%)'],
        col_format='rrrrrrrrrr',
        caption="Complete experimental results for all 36 combinations of $p'$, $c_{esc}$, and $c_{rej}$. $ETC_{base}$ is the static-baseline cost ($i=50$, $f=0.10$). $AOQ^*$ is the outgoing quality at the scenario's true $p'$, while $AOQL^*$ is the worst-case outgoing quality of the optimal plan across all $p'$. In the escape-cheap regime the economic optimum tolerates a high $AOQL^*$ (light screening), whereas in the high-risk regime it drives $AOQL^* \\to 0$ via full inspection. A practitioner with a binding AOQL target can read off which scenarios satisfy it.",
        label="app:fulltable",
        scale=0.72
    )

    with open(os.path.join(base_path, 'results_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_rep)
    with open(os.path.join(base_path, 'results_table_full.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_full)

    print(f"LaTeX tables saved: results_table.tex ({len(df_rep)} rows) + "
          f"results_table_full.tex ({len(df)} rows, with AOQL column)")


if __name__ == "__main__":
    generate_table()
