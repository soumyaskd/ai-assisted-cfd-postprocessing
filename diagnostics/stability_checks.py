import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "solver"
REPORT_DIR = BASE_DIR / "report"

REPORT_DIR.mkdir(exist_ok=True)

residuals_path = DATA_DIR / "residuals.csv"
forces_path = DATA_DIR / "forces.csv"

if not residuals_path.exists():
    print(f"Residuals file not found at: {residuals_path}")
    sys.exit(1)

if not forces_path.exists():
    print(f"Forces file not found at: {forces_path}")
    sys.exit(1)

# --------------------------------------------------
# Load data
# --------------------------------------------------
res_df = pd.read_csv(residuals_path)
force_df = pd.read_csv(forces_path)

residual_cols = [c for c in res_df.columns if c != "iteration"]
force_cols = [c for c in force_df.columns if c != "iteration"]

# --------------------------------------------------
# Residual health checks
# --------------------------------------------------
residual_floor = 1e-7
residual_pass = True
residual_notes = []

for col in residual_cols:
    final_res = res_df[col].iloc[-1]
    if final_res > residual_floor:
        residual_pass = False
        residual_notes.append(
            f"{col}: final residual {final_res:.2e} above tolerance"
        )

# --------------------------------------------------
# Refined force drift logic (TAIL-ONLY)
# --------------------------------------------------
def tail_mean_drift(signal, tail_fraction=0.2):
    n = len(signal)
    tail_start = int((1 - tail_fraction) * n)
    tail = signal[tail_start:]

    mid = len(tail) // 2
    mean_1 = np.mean(tail[:mid])
    mean_2 = np.mean(tail[mid:])

    denom = max(abs(mean_1), 1e-6)
    return abs(mean_2 - mean_1) / denom

force_pass = True
force_notes = []

for col in force_cols:
    drift = tail_mean_drift(force_df[col])

    if drift > 1e-3:
        force_pass = False
        force_notes.append(
            f"{col}: tail mean drift {drift:.2e} exceeds threshold"
        )

# --------------------------------------------------
# Cross-consistency logic
# --------------------------------------------------
verdict = "PASS"
reasons = []

if not residual_pass and not force_pass:
    verdict = "FAIL"
    reasons.append("Neither residuals nor forces converged")

elif residual_pass and not force_pass:
    verdict = "FAIL"
    reasons.append("Residuals converged but forces show late-stage drift")

elif not residual_pass and force_pass:
    verdict = "WARNING"
    reasons.append("Forces converged but residuals did not fully converge")

else:
    reasons.append("Residuals and forces converged consistently")

# --------------------------------------------------
# Console output
# --------------------------------------------------
print("\nFINAL SOLVER STABILITY VERDICT (REFINED)")
print("--------------------------------------")
print(f"Verdict: {verdict}")

for r in reasons:
    print(" -", r)

if residual_notes:
    print("\nResidual issues:")
    for note in residual_notes:
        print(" *", note)

if force_notes:
    print("\nForce issues:")
    for note in force_notes:
        print(" *", note)

# --------------------------------------------------
# Write report summary
# --------------------------------------------------
report_path = REPORT_DIR / "solver_health.md"

with open(report_path, "w") as f:
    f.write("# Solver Stability Assessment\n\n")
    f.write(f"**Final verdict:** {verdict}\n\n")

    f.write("## Summary\n")
    for r in reasons:
        f.write(f"- {r}\n")

    if residual_notes:
        f.write("\n## Residual Issues\n")
        for note in residual_notes:
            f.write(f"- {note}\n")

    if force_notes:
        f.write("\n## Force Issues\n")
        for note in force_notes:
            f.write(f"- {note}\n")

    if verdict == "PASS":
        f.write(
            "\nThe CFD solution is numerically stable and physically converged. "
            "Late-stage force histories show negligible drift, and residuals "
            "have decayed to the solver tolerance floor. The solution is "
            "considered trustworthy.\n"
        )

print("\nSolver health report written to:", report_path)
print("Refined stability checks COMPLETE.\n")
