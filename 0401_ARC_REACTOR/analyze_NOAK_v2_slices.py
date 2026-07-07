#!/usr/bin/env python3
# ============================================================
# NOAK Deployment Overplot Tool
# ============================================================

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "legend.frameon": False,
})

ARC_POWER_MWE = 190.0

# ============================================================
# USER INPUT
# ============================================================

MODE = "CAGR_FIXED"   # or "BE_FIXED"
# MODE = "BE_FIXED"   # or "BE_FIXED"

CAGR_FIXED = 4
BE_LIST = [1,2,4,6,8]

BE_FIXED = 2
CAGR_LIST = [0,2,4,6,8]

# ============================================================
# Helper: get time-dependent fleet power
# ============================================================

def get_fleet_curve(be, cagr):

    sqlite_file = f"RUNS/ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF/out_ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF.sqlite"

    if not os.path.exists(sqlite_file):
        return None, None

    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    cur.execute("""
    SELECT Time
    FROM Transactions
    WHERE Commodity = 'built_flag'
    ORDER BY Time
    """)

    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        return None, None

    build_times = np.array([r[0] for r in rows], dtype=float)
    build_times = build_times/12
    n_arc = np.arange(1, len(build_times)+1)
    fleet_power = n_arc * ARC_POWER_MWE / 1e3

    return build_times, fleet_power

# ============================================================
# CASE 1 — CAGR fixed, multiple Be
# ============================================================

if MODE == "CAGR_FIXED":

    plt.figure()

    markers = ['o','s','^','D','v','P','X']
    colors = plt.cm.tab10.colors

    for idx, be in enumerate(BE_LIST):

        t, p = get_fleet_curve(be, CAGR_FIXED)
        if t is None:
            continue

        plt.step(t, p,
                 where='post',
                 #marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 label=f"Be = {be} t/mo")

    plt.xlabel("Time (year)")
    plt.ylabel("Fleet Power (GWe)")
    plt.title(f"Deployment (CAGR = {CAGR_FIXED}%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.ylim([0,25])
    plt.show()

# ============================================================
# CASE 2 — Be fixed, multiple CAGR
# ============================================================

if MODE == "BE_FIXED":

    plt.figure()

    markers = ['o','s','^','D','v','P','X']
    colors = plt.cm.tab10.colors

    for idx, cagr in enumerate(CAGR_LIST):

        t, p = get_fleet_curve(BE_FIXED, cagr)
        if t is None:
            continue

        plt.step(t, p,
                 where='post',
                 #marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 label=f"CAGR = {cagr}%")

    plt.xlabel("Time (year)")
    plt.ylabel("Fleet Power (GWe)")
    plt.title(f"Deployment (Be = {BE_FIXED} t/mo)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.ylim([0,25])
    plt.show()