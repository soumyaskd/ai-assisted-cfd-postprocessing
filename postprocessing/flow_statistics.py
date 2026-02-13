import pandas as pd
import numpy as np

# Load CFD data
#df = pd.read_csv("data\flow_fields\large_cfd_dataset.csv")

try:
    df = pd.read_csv("data\flow_fields\large_cfd_dataset.csv")
except FileNotFoundError:
    print("CFD data file not found. Check the data folder.")
    import sys
    sys.exit(1)
    #exit()

# Basic metrics
mean_velocity = np.mean(df["velocity"])
max_velocity = np.max(df["velocity"])
min_velocity = np.min(df["velocity"])
mean_pressure = np.mean(df["pressure"])
Velocity_range = max_velocity - min_velocity
velocity_std = np.std(df["velocity"])  # Standard deviation of velocity - unsteadiness measure - local tubulence strength
velocity_variance = np.var(df["velocity"]) # Variance of velocity - another unsteadiness measure - energy content of fluctuations
rms_velocity = np.sqrt(np.mean(np.square(df["velocity"]))) # Root mean square velocity - 
#turbulence intensity measure (Energy content of velocity fluctuations)) - turbulence proxy - kinetic energy indicator

pressure_range = np.max(df["pressure"]) - np.min(df["pressure"]) # Solver health metric - 
#large pressure range may indicate convergence issues or flow separation

pressure_gradient = np.gradient(df["pressure"], df["time"])   #rate of change
mean_pressure_gradient = np.mean(np.abs(pressure_gradient))

acceleration = np.gradient(df["velocity"], df["time"])
mean_acceleration = np.mean(np.abs(acceleration))

velocity_cov = velocity_std / mean_velocity # Coefficient of variation for velocity - relative variability measure
#How big are the speed changes compared to average speed? - Normalized turbulence measure





print("CFD Summary Metrics")
print("-------------------")
print(f"Mean velocity: {mean_velocity:.2f} m/s")
print(f"Max velocity: {max_velocity:.2f} m/s")
print(f"Mean pressure: {mean_pressure:.2f} Pa")
print(f"Velocity range: {Velocity_range:.2f} m/s")

print("Advanced CFD Metrics")
print("-------------------")
print(f"Velocity standard deviation: {velocity_std:.2f} m/s")
print(f"Velocity variance: {velocity_variance:.3f} (m/s)²")
print(f"RMS velocity: {rms_velocity:.2f} m/s")
print(f"Pressure range: {pressure_range:.2f} Pa")
print(f"PRessure gradient: {pressure_gradient}")
print(f"Mean pressure gradient: {mean_pressure_gradient:.2f} Pa/s")
print(f"Mean acceleration: {mean_acceleration:.2f} m/s²")
print(f"Velocity coefficient of variation: {velocity_cov:.2f}")
