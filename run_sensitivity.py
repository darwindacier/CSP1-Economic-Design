import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csp_model import CSP1EconomicModel, optimize_csp1

def run_sensitivity():
    p_values = [0.01, 0.02, 0.05, 0.10]
    c_esc_values = [20, 50, 100]
    c_rej_values = [100, 500, 1000]
    
    results = []
    
    print("Running sensitivity analysis (36 combinations)...")
    for p in p_values:
        for c_esc in c_esc_values:
            for c_rej in c_rej_values:
                (i_opt, f_opt), cost = optimize_csp1(p, c_esc=c_esc, c_rej=c_rej)
                
                # Baseline comparison (i=50, f=0.10)
                m_static = CSP1EconomicModel(i=50, f=0.10, p_prime=p)
                cost_static = m_static.expected_total_cost(c_esc=c_esc, c_rej=c_rej)
                
                reduction = (1 - cost / cost_static) * 100 if cost_static > 0 else 0
                
                results.append({
                    'p': p,
                    'c_esc': c_esc,
                    'c_rej': c_rej,
                    'i_opt': i_opt,
                    'f_opt': round(f_opt, 4),
                    'cost_opt': round(cost, 2),
                    'cost_static': round(cost_static, 2),
                    'reduction_%': round(reduction, 2)
                })
                print(f"p={p}, c_esc={c_esc}, c_rej={c_rej} -> i*={i_opt}, f*={f_opt:.4f}, Cost*={cost:.2f}, Red={reduction:.2f}%")
    
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensitivity_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Sensitivity results saved to {output_path}")

if __name__ == "__main__":
    run_sensitivity()
