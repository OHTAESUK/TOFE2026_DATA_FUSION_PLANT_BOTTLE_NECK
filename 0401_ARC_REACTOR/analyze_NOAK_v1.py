#!/usr/bin/env python3
# ============================================================
# NOAK 2D Sweep Analysis
# Heatmap + Slice plots
# ============================================================

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# USER SETTINGS
# ============================================================
TARGET_YEARS = 50
TARGET_MONTH = TARGET_YEARS * 12

# ============================================================
# PLOT STYLE (Global)
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "legend.frameon": False,
})

BE_RANGE = range(1, 11)
CAGR_RANGE = range(0, 11)

ARC_POWER_MWE = 190.0

fleet_power = np.zeros((len(BE_RANGE), len(CAGR_RANGE)))

# ============================================================
# Collect fleet power at user-defined year
# ============================================================

for i, be in enumerate(BE_RANGE):
    for j, cagr in enumerate(CAGR_RANGE):

        case = f"RUNS/ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF"
        sqlite_file = f"{case}/out_ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF.sqlite"

        if not os.path.exists(sqlite_file):
            continue

        conn = sqlite3.connect(sqlite_file)
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*)
            FROM Transactions
            WHERE Commodity = 'built_flag'
            AND Time <= ?
        """, (TARGET_MONTH,))

        result = cur.fetchone()
        conn.close()

        if result is None:
            continue

        n_arc = result[0]

        fleet_power[i, j] = n_arc * ARC_POWER_MWE / 1e3

# ============================================================
# HEATMAP
# ============================================================

plt.figure(figsize=(8,6))

im = plt.imshow(
    fleet_power,
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

plt.colorbar(im, label="Fleet Power (GWe)")

plt.xticks(range(len(CAGR_RANGE)), CAGR_RANGE)
plt.yticks(range(len(BE_RANGE)), BE_RANGE)

# ============================================================
# Add cell values
# ============================================================

for i, be in enumerate(BE_RANGE):
    for j, cagr in enumerate(CAGR_RANGE):

        value = fleet_power[i, j]

        if np.isfinite(value):
            plt.text(
                j,                  # x index = CAGR
                i,                  # y index = Be
                f"{value:.1f}",     # value format
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

plt.xlabel("CAGR (%)")
plt.ylabel("Initial Be Supply (t/month)")
plt.title(f"Fleet Power after {TARGET_YEARS} Years")

plt.tight_layout()
plt.show()

# ============================================================
# SLICE 1 — CAGR fixed
# ============================================================

plt.figure()

CAGR_FIXED = 5
j = CAGR_FIXED

be_vals = []
power_vals = []

for i, be in enumerate(BE_RANGE):
    be_vals.append(be)
    power_vals.append(fleet_power[i, j])

plt.plot(be_vals,
         power_vals,
         marker='o',
         linewidth=2.5)

plt.xlabel("Be Supply (t/month)")
plt.ylabel("Fleet Power (GWe)")
plt.title(f"Fleet Power vs Be (CAGR = {CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# SLICE 2 — Be fixed
# ============================================================

BE_FIXED = 5

plt.figure()

i = BE_FIXED - 1
plt.plot(list(CAGR_RANGE), fleet_power[i, :], marker='o')

plt.xlabel("CAGR (%)")
plt.ylabel("Fleet Power (GWe)")
plt.title(f"Fleet Power vs CAGR (Be = {BE_FIXED} t/month)")
plt.grid(True)
plt.show()

print("\n[INFO] 2D NOAK analysis complete.\n")