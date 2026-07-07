#!/usr/bin/env python3
# ============================================================
# NOAK v5: Resource CSV verification plotter
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = "NOAK_ARC_resource_requirements_monthly.csv"

BE_FIXED = 2          # t/month
CAGR_FIXED = 4        # %

# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.3,
    "legend.frameon": False,
})

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

case_df = df[
    (df["be_t_per_month"] == BE_FIXED) &
    (df["cagr_percent"] == CAGR_FIXED)
].copy()

if case_df.empty:
    raise ValueError(
        f"No data found for Be={BE_FIXED} t/month, CAGR={CAGR_FIXED}%"
    )

case_df = case_df.sort_values("time_month")

print("--------------------------------------------------")
print(f"[INFO] Loaded case: Be={BE_FIXED} t/month, CAGR={CAGR_FIXED}%")
print(f"[INFO] Rows: {len(case_df)}")
print(f"[INFO] Final ARC built: {case_df['n_arc_built'].iloc[-1]}")
print(f"[INFO] Final fleet power: {case_df['fleet_power_gwe'].iloc[-1]:.2f} GWe")
print(f"[INFO] Final Li-90 startup: {case_df['li90_startup_cum_t'].iloc[-1]:,.2f} t")
print(f"[INFO] Final Li-90 operation: {case_df['li90_operation_cum_t'].iloc[-1]:,.2f} t")
print(f"[INFO] Final Li-90 total: {case_df['li90_total_cum_t'].iloc[-1]:,.2f} t")
print(f"[INFO] Final REBCO: {case_df['rebco_cum_km'].iloc[-1]:,.0f} km")
print("--------------------------------------------------")

t = case_df["time_year"]

# ============================================================
# PLOT 1: ARC DEPLOYMENT
# ============================================================

plt.figure(figsize=(8, 5))

plt.step(
    t,
    case_df["n_arc_built"],
    where="post",
    label="ARC built"
)

plt.xlabel("Time (year)")
plt.ylabel("Number of ARC plants")
plt.title(f"ARC Deployment (Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2: FLEET POWER
# ============================================================

plt.figure(figsize=(8, 5))

plt.step(
    t,
    case_df["fleet_power_gwe"],
    where="post",
    label="Fleet power"
)

plt.xlabel("Time (year)")
plt.ylabel("Fleet Power (GWe)")
plt.title(f"Fleet Power (Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 3: Li-90 REQUIREMENT
# ============================================================

plt.figure(figsize=(8, 5))

plt.step(
    t,
    case_df["li90_startup_cum_t"],
    where="post",
    label="Startup Li-90"
)

plt.plot(
    t,
    case_df["li90_operation_cum_t"],
    linestyle="--",
    label="Operation Li-90"
)

plt.plot(
    t,
    case_df["li90_total_cum_t"],
    linewidth=2.8,
    label="Total Li-90"
)

plt.xlabel("Time (year)")
plt.ylabel("Cumulative Li-90 Requirement (t)")
plt.title(f"Li-90 Requirement (Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 4: REBCO REQUIREMENT
# ============================================================

plt.figure(figsize=(8, 5))

plt.step(
    t,
    case_df["rebco_cum_km"],
    where="post",
    label="REBCO demand"
)

plt.xlabel("Time (year)")
plt.ylabel("Cumulative REBCO Requirement (km)")
plt.title(f"REBCO Requirement (Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 5: Li vs REBCO normalized check
# ============================================================

plt.figure(figsize=(8, 5))

li_norm = case_df["li90_total_cum_t"] / case_df["li90_total_cum_t"].max()
rebco_norm = case_df["rebco_cum_km"] / case_df["rebco_cum_km"].max()

plt.plot(t, li_norm, label="Li-90 total, normalized")
plt.step(t, rebco_norm, where="post", label="REBCO, normalized")

plt.xlabel("Time (year)")
plt.ylabel("Normalized cumulative requirement")
plt.title("Li-90 vs REBCO Growth Check")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("[INFO] v5 verification plots complete.")