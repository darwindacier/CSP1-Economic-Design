import pandas as pd
from csp_model import CSP1EconomicModel

# Verify a few specific scenarios against the CSV
df = pd.read_csv('sensitivity_results.csv')

# Check Proposition 1: saturated cases should equal ETC_inf = c_ins*Q + c_rew*p'*Q
Q, c_ins, c_rew = 1000, 1.0, 20.0
print('=== Proposition 1 verification ===')
for p in [0.05, 0.10]:
    etc_inf = c_ins*Q + c_rew*p*Q
    rows = df[(df['p']==p) & (df['i_opt']>=400)]
    for _, r in rows.iterrows():
        match = 'MATCH' if abs(r['cost_opt'] - etc_inf) < 0.01 else 'MISMATCH'
        print(f"p={p}: ETC_inf={etc_inf:.2f}, CSV={r['cost_opt']:.2f} -> {match}")

# Check switching rule consistency
print()
print('=== Switching rule consistency ===')
for _, r in df.iterrows():
    p, c_esc, c_rew = r['p'], r['c_esc'], 20.0
    threshold = c_ins / (c_esc - c_rew) if c_esc > c_rew else float('inf')
    saturated = r['i_opt'] >= 400
    pred_saturated = p > threshold
    if saturated != pred_saturated and abs(p - threshold) > 0.005:
        print(f"  ANOMALY: p={p}, c_esc={c_esc}: i*={r['i_opt']}, threshold={threshold:.4f}, pred={pred_saturated}, actual={saturated}")
print("  (Anomalies expected near threshold due to c_rej influence)")

# Verify baseline costs
print()
print('=== Baseline cost verification ===')
for _, r in df.head(3).iterrows():
    m = CSP1EconomicModel(i=50, f=0.10, p_prime=r['p'], lot_size=1000)
    etc_base = m.expected_total_cost(c_ins=1.0, c_esc=r['c_esc'], c_rej=r['c_rej'], c_rew=20.0)
    csv_base = r['cost_static']
    print(f"p={r['p']}, c_esc={r['c_esc']}, c_rej={r['c_rej']}: computed={etc_base:.2f}, CSV={csv_base:.2f}, diff={abs(etc_base-csv_base):.4f}")
