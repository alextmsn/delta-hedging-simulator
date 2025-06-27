import numpy as np
import math

def simulate_gbm_path(S0, mu, sigma, T, steps):
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        z = np.random.normal()
        S_prev = prices[-1]
        S_new = S_prev * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
        prices.append(S_new)
    return prices

def simulate_heston_path(S0, v0, r, kappa, theta, xi, rho, T, steps):
    dt = T / steps
    prices = [S0]
    vols = [v0]
    S = S0
    v = v0
    for _ in range(steps):
        z1, z2 = np.random.normal(size=2)
        dw1 = z1
        dw2 = rho * z1 + math.sqrt(1 - rho**2) * z2
        v = max(v + kappa * (theta - v) * dt + xi * math.sqrt(v * dt) * dw2, 1e-8)
        S = S * math.exp((r - 0.5 * v) * dt + math.sqrt(v * dt) * dw1)
        prices.append(S)
        vols.append(v)
    return prices, vols
