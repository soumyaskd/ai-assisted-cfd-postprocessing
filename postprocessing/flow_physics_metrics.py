import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --------------------------------------------------
# Project paths (robust & portable)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "flow_fields"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "large_cfd_dataset.csv"

if not csv_path.exists():
    print(f"CFD data file not found at: {csv_path}")
    sys.exit(1)

# --------------------------------------------------
# Load CFD flow-field data
# --------------------------------------------------
df = pd.read_csv(csv_path)

# --------------------------------------------------
# Basic flow metrics
# --------------------------------------------------
mean_velocity = np.mean(df["velocity"])
max_velocity = np.max(df["velocity"])
min_velocity = np.min(df["velocity"])
velocity_range = max_velocity - min_velocity

mean_pressure = np.mean(df["pressure"])
pressure_range = np.max(df["pressure"]) - np.min(df["pressure"])

velocity_std = np.std(df["velocity"])
velocity_variance = np.var(df["velocity"])
rms_velocity = np.sqrt(np.mean(df["velocity"] ** 2))

velocity_cov = velocity_std / mean_velocity

# --------------------------------------------------
# Time-derivative based metrics
# --------------------------------------------------
pressure_gradient = np.gradient(df["pressure"], df["time"])
abs_pressure_gradient = np.abs(pressure_gradient)
mean_pressure_gradient = np.mean(abs_pressure_gradient)

acceleration = np.gradient(df["velocity"], df["time"])
mean_acceleration = np.mean(np.abs(acceleration))

cumulative_pressure_drop = df["pressure"].iloc[0] - df["pressure"]

# --------------------------------------------------
# Normalized signals (z-score)
# --------------------------------------------------
v_norm = (df["velocity"] - df["velocity"].mean()) / df["velocity"].std()
p_norm = (df["pressure"] - df["pressure"].mean()) / df["pressure"].std()

# --------------------------------------------------
# Rolling statistics (local unsteadiness)
# --------------------------------------------------
window_size = 3
rolling_pressure_std = df["pressure"].rolling(window=window_size).std()
rolling_velocity_std = df["velocity"].rolling(window=window_size).std()

# --------------------------------------------------
# Console output (engineering summary)
# --------------------------------------------------
print("\nCFD FLOW PHYSICS SUMMARY")
print("------------------------")
print(f"Mean velocity              : {mean_velocity:.2f} m/s")
print(f"Velocity range             : {velocity_range:.2f} m/s")
print(f"Velocity std deviation     : {velocity_std:.2f} m/s")
print(f"Velocity CoV               : {velocity_cov:.3f}")
print(f"RMS velocity               : {rms_velocity:.2f} m/s")

print("\nPRESSURE BEHAVIOR")
print("-----------------")
print(f"Mean pressure              : {mean_pressure:.2f} Pa")
print(f"Pressure range             : {pressure_range:.2f} Pa")
print(f"Mean |dP/dt|               : {mean_pressure_gradient:.2f} Pa/s")

print("\nUNSTEADINESS INDICATORS")
print("----------------------")
print(f"Mean |acceleration|        : {mean_acceleration:.3f} m/s²")

print("\nNORMALIZATION CHECK")
print("-------------------")
print(f"Velocity norm mean/std     : {v_norm.mean():.2f} / {v_norm.std():.2f}")
print(f"Pressure norm mean/std     : {p_norm.mean():.2f} / {p_norm.std():.2f}")

print("\nRolling pressure std (sample):")
print(rolling_pressure_std.dropna().head(5))

print("\nRolling velocity std (sample):")
print(rolling_velocity_std.dropna().head(5))

# --------------------------------------------------
# Plotting section
# --------------------------------------------------

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

save_plot(df["time"], df["velocity"],
          "Time (s)", "Velocity (m/s)",
          "Velocity vs Time", "velocity_vs_time.png")

save_plot(df["time"], df["pressure"],
          "Time (s)", "Pressure (Pa)",
          "Pressure vs Time", "pressure_vs_time.png")

save_plot(df["time"], pressure_gradient,
          "Time (s)", "dP/dt (Pa/s)",
          "Pressure Gradient vs Time", "pressure_gradient_vs_time.png")

save_plot(df["time"], acceleration,
          "Time (s)", "Acceleration (m/s²)",
          "Acceleration vs Time", "acceleration_vs_time.png")

save_plot(df["pressure"], df["velocity"],
          "Pressure (Pa)", "Velocity (m/s)",
          "Pressure vs Velocity", "pressure_vs_velocity.png")

save_plot(df["time"], abs_pressure_gradient,
          "Time (s)", "|dP/dt| (Pa/s)",
          "Absolute Pressure Gradient vs Time", "abs_pressure_gradient_vs_time.png")

plt.figure()
plt.plot(df["time"], v_norm, label="Velocity (normalized)")
plt.plot(df["time"], p_norm, label="Pressure (normalized)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized value")
plt.title("Normalized Velocity and Pressure vs Time")
plt.legend()
plt.grid(True)
plt.savefig(PLOTS_DIR / "normalized_velocity_pressure_vs_time.png")
plt.close()

save_plot(df["time"], rolling_pressure_std,
          "Time (s)", "Rolling Pressure Std (Pa)",
          "Rolling Pressure Std vs Time", "rolling_pressure_variance_vs_time.png")

save_plot(df["time"], rolling_velocity_std,
          "Time (s)", "Rolling Velocity Std (m/s)",
          "Rolling Velocity Std vs Time", "rolling_velocity_variance_vs_time.png")

print("\nPlots saved to:", PLOTS_DIR)
print("Flow physics post-processing COMPLETE.\n")
