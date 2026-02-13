import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "solver"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "residuals.csv"

if not csv_path.exists():
    print(f"Residuals file not found at: {csv_path}")
    sys.exit(1)

# --------------------------------------------------
# Load residual data
# --------------------------------------------------
df = pd.read_csv(csv_path)

iteration = df["iteration"]

# Residual columns (exclude iteration)
residual_cols = [c for c in df.columns if c != "iteration"]

# --------------------------------------------------
# Diagnostic metrics
# --------------------------------------------------

def residual_decay_rate(residual):
    """
    Linear slope of log10(residual) vs iteration.
    More negative = better convergence.
    """
    log_r = np.log10(residual)
    slope = np.polyfit(iteration, log_r, 1)[0]
    return slope


def stagnation_index(residual, window=50):
    """
    Detects pseudo-convergence.
    Low std + high mean => solver stuck.
    """
    tail = residual[-window:]
    return np.std(tail) / np.mean(tail)


# --------------------------------------------------
# Compute diagnostics
# --------------------------------------------------
results = {}

for col in residual_cols:
    res = df[col].values

    decay = residual_decay_rate(res)
    stagnation = stagnation_index(res)

    results[col] = {
        "final_residual": res[-1],
        "decay_rate": decay,
        "stagnation_index": stagnation
    }

# --------------------------------------------------
# Console report (this is the money part)
# --------------------------------------------------
print("\nSOLVER RESIDUAL DIAGNOSTICS")
print("---------------------------")

for eqn, metrics in results.items():
    print(f"\nEquation: {eqn}")
    print(f"  Final residual      : {metrics['final_residual']:.2e}")
    print(f"  Decay rate (slope)  : {metrics['decay_rate']:.3e}")
    print(f"  Stagnation index    : {metrics['stagnation_index']:.3e}")

    if metrics["decay_rate"] > -1e-4:
        print("  ⚠ Residual decay is weak (possible pseudo-convergence)")

    if metrics["stagnation_index"] < 1e-3:
        print("  ⚠ Residual may be numerically stuck")

# --------------------------------------------------
# Residual plots (log scale, solver-style)
# --------------------------------------------------
plt.figure(figsize=(8, 6))

for col in residual_cols:
    plt.semilogy(iteration, df[col], label=col)

plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.title("Solver Residual Convergence History")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(PLOTS_DIR / "residual_convergence.png")
plt.close()

print("\nResidual convergence plot saved to:", PLOTS_DIR)
print("Residual analysis COMPLETE.\n")
