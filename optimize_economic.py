# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from csp_model import CSP1EconomicModel, optimize_csp1

def run_comparison():
    p_values = np.linspace(0.005, 0.10, 20)
    costs_opt = []
    costs_static = []
    i_static, f_static = 50, 0.10
    for p in p_values:
        params, cost = optimize_csp1(p)
        costs_opt.append(cost)
        m_static = CSP1EconomicModel(i=i_static, f=f_static, p_prime=p)
        costs_static.append(m_static.expected_total_cost())
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, costs_opt, "o-", label="Optimized (i*, f*)")
    plt.plot(p_values, costs_static, "s--", label=f"Static (i={i_static}, f={f_static})")
    plt.xlabel("Incoming Defect Rate (p)")
    plt.ylabel("Expected Total Cost per Lot ($)")
    plt.title("Economic Comparison: Optimized vs Static CSP-1")
    plt.legend()
    plt.grid(True)
    plt.savefig("cost_comparison.png", dpi=300, facecolor='white')
    plt.savefig("cost_comparison.eps", dpi=300, facecolor='white')
    print("Comparison plot saved as cost_comparison.png (300 dpi, RGB) + EPS")

if __name__ == "__main__":
    run_comparison()

