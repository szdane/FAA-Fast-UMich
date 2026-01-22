import math
import matplotlib.pyplot as plt
import numpy as np

def fuel_flow(c1, c2, c3, scale, x):
    """
    Compute fuel flow given parameters and throttle ratio x.
    Formula:
        f(x) = c1 - exp(-c2 * (x * exp(c3*x) - log(c1)/c2)) * scale
    """
    return c1 - math.exp(-c2 * (x * math.exp(c3 * x) - math.log(c1) / c2)) * scale

def fuel_flow_sl_poly(cff1, cff2, cff3, throttle):
    """
    Sea-level fuel flow polynomial based on throttle ratio x = T/T0:
        f(x) = C_ff3 * x^3 + C_ff2 * x^2 + C_ff1 * x
    """
    x = throttle
    return cff3 * x**3 + cff2 * x**2 + cff1 * x

# Example usage
if __name__ == "__main__":
    c1, c2, c3, scale = 1.8497366218576496,2.0668722638298687,1.5510804722001057, 1.85
    # quick sanity print
    for x in [0.15]:
        print(f"x={x:.1f}, fuel_flow={fuel_flow(c1, c2, c3, scale, x):.4f}")

    # Plot fuel flow vs throttle (x in [0, 1])
    xs = np.linspace(0.0, 1.0, 200)
    ys = [fuel_flow(c1, c2, c3, scale, xi) for xi in xs]

    # Fit the polynomial f(x) = c1*x + c2*x^2 + c3*x^3 (no constant term)
    A = np.column_stack([xs, xs**2, xs**3])
    c_fit, *_ = np.linalg.lstsq(A, ys, rcond=None)
    cff1_fit, cff2_fit, cff3_fit = c_fit.tolist()
    ys_poly = fuel_flow_sl_poly(cff1_fit, cff2_fit, cff3_fit, xs)

    plt.figure(figsize=(6, 3))
    plt.plot(xs, ys, label="exp model", color="tab:red")
    plt.plot(xs, ys_poly, label="poly (SL) fit", color="tab:blue", linestyle="--")
    plt.xlabel("throttle (T/T0)")
    plt.ylabel("fuel flow (kg/s)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()