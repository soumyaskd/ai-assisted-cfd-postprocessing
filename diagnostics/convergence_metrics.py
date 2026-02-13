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

csv_path = DATA_DIR / "forces.csv"

if not csv_path.exists():
    print(f"Forces file not found at: {csv_path}")
    sys.exit(1)

# --------------------------------------------------
# Load force history
# --------------------------------------------------
df = pd.read_csv(csv_path)

iteration = df["iteration"]

force_cols = [c for c in df.columns if c != "iteration"]

# --------------------------------------------------
# Convergence metrics
# --------------------------------------------------

def rolling_std(signal, window=100):
    return signal.rolling(window=window).std()


def mean_drift(signal):
    mid = len(signal) // 2
    mean_1 = np.mean(signal[:mid])
    mean_2 = np.mean(signal[mid:])
    return abs(mean_2 - mean_1) / abs(mean_1)


# --------------------------------------------------
# Compute metrics
# --------------------------------------------------
results = {}

for force in force_cols:
    signal = df[force]

    std_tail = rolling_std(signal).iloc[-1]
    drift = mean_drift(signal)

    results[force] = {
        "final_value": signal.iloc[-1],
        "rolling_std_tail": std_tail,
        "mean_drift": drift
    }

# --------------------------------------------------
# Console report (this is critical)
# --------------------------------------------------
print("\nFORCE CONVERGENCE DIAGNOSTICS")
print("----------------------------")

for force, metrics in results.items():
    print(f"\nForce: {force}")
    print(f"  Final value        : {metrics['final_value']:.4f}")
    print(f"  Rolling STD (tail) : {metrics['rolling_std_tail']:.4e}")
    print(f"  Mean drift         : {metrics['mean_drift']:.4e}")

    if metrics["mean_drift"] > 1e-3:
        print("  ⚠ Force still drifting → physical convergence NOT achieved")

    if metrics["rolling_std_tail"] > 1e-2:
        print("  ⚠ Significant unsteadiness remains")

# --------------------------------------------------
# Plot: force convergence history
# --------------------------------------------------
plt.figure(figsize=(8, 6))

for force in force_cols:
    plt.plot(iteration, df[force], label=force)

plt.xlabel("Iteration")
plt.ylabel("Force")
plt.title("Force Convergence History")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOTS_DIR / "force_convergence.png")
plt.close()

print("\nForce convergence plot saved to:", PLOTS_DIR)
print("Force convergence analysis COMPLETE.\n")
