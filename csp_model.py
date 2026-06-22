"""
CSP-1 Continuous Sampling Plan with Markov Analysis and Economic Model.
Supports 4 cost components: inspection, escape, rejection, and rework. 
"""
import numpy as np
from scipy.optimize import minimize_scalar

class CSP1EconomicModel:
    def __init__(self, i=50, f=0.10, p_prime=0.02, lot_size=1000):     
        self.i = int(max(1, round(i)))
        self.f = float(max(0.0001, min(1.0, f)))
        self.p_prime = float(p_prime)
        self.lot_size = lot_size
        self._compute_steady_state()

    def _compute_steady_state(self):
        """Steady-state phase fractions via a regenerative (renewal-reward) cycle.

        A CSP-1 regeneration cycle is a 100%-inspection phase of expected length
        ``u`` units followed by a fractional-sampling phase of expected length
        ``v`` units, where (with q = 1 - p' and clearance i)
            u = (1 - q^i) / (p' q^i),   v = 1 / (f p').
        The phase fractions are pi1 = u/L and pi2 = v/L with cycle length L = u + v.
        Unlike a naive DTMC balance, pi2 depends on the sampling fraction f, so the
        resulting AFI = pi1 + pi2*f = f / (f + (1-f) q^i) coincides exactly with the
        classical CSP-1 expression of Dodge (1943).
        """
        p = self.p_prime
        q = 1.0 - p
        f = self.f
        qi = q ** self.i

        if p <= 0.0:
            # No defects ever: after the initial clearance, the process samples
            # forever, so the cycle degenerates to the sampling phase.
            self.pi_100 = 0.0
            self.pi_frac = 1.0
            self.afi = f
            self.cycle_length = float('inf')
        elif qi <= 1e-300 or (p * qi) <= 0.0:
            # i -> infinity regime: the clearance is essentially never achieved, so
            # the process stays in 100% inspection (u -> infinity, pi2 -> 0).
            self.pi_100 = 1.0
            self.pi_frac = 0.0
            self.afi = 1.0
            self.cycle_length = float('inf')
        else:
            u = (1.0 - qi) / (p * qi)   # expected 100%-inspection phase length (units)
            v = 1.0 / (f * p)           # expected fractional-sampling phase length (units)
            L = u + v
            self.pi_100 = u / L
            self.pi_frac = v / L
            # AFI = pi1 + pi2*f  ==  f / (f + (1-f) q^i)   (exact simplification)
            self.afi = f / (f + (1.0 - f) * qi)
            self.cycle_length = L

        # AOQ: Average Outgoing Quality
        self.aoq = p * (1.0 - self.afi)

        # Expected metrics per lot
        self.n_inspected = self.afi * self.lot_size
        self.n_escaped = self.aoq * self.lot_size
        self.n_found = p * self.afi * self.lot_size

        # Expected number of transitions to 100% mode per lot. By the renewal
        # argument this equals Q / L; the form pi2*f*p'*Q is algebraically identical
        # and was already correct in the original model.
        self.transition_rate = self.pi_frac * f * p * self.lot_size

    def expected_total_cost(self, c_ins=1.0, c_esc=50.0, c_rej=500.0, c_rew=20.0):
        """Compute expected total cost based on the 4 components."""
        cost_ins = c_ins * self.n_inspected
        cost_esc = c_esc * self.n_escaped
        # Note: c_rej is applied to the expected number of transitions to 100% mode
        cost_rej = c_rej * self.transition_rate
        cost_rew = c_rew * self.n_found
        return cost_ins + cost_esc + cost_rej + cost_rew

def optimize_csp1(p_prime, lot_size=1000,
                             c_ins=1.0, c_esc=50.0, c_rej=500.0, c_rew=20.0,
                             i_bounds=(1, 500), f_bounds=(0.01, 1.0)):
    """Minimize ETC using grid search for i and continuous optimization for f."""
    best_cost = float('inf')
    best_params = (50, 0.1)

    for i in range(i_bounds[0], i_bounds[1] + 1):
        def objective(f):
            m = CSP1EconomicModel(i=i, f=f, p_prime=p_prime, lot_size=lot_size)
            return m.expected_total_cost(c_ins, c_esc, c_rej, c_rew)

        res = minimize_scalar(objective, bounds=f_bounds, method='bounded')
        if res.success and res.fun < best_cost:
            best_cost = res.fun
            best_params = (i, res.x)

    return best_params, best_cost

if __name__ == "__main__":
    p = 0.02
    (i_opt, f_opt), cost = optimize_csp1(p)
    print(f"Optimal (p={p}): i={i_opt}, f={f_opt:.4f}, Cost=${cost:.2f}")
