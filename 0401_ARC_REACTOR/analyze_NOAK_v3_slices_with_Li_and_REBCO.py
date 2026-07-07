#!/usr/bin/env python3
# ============================================================
# NOAK Deployment + Resource Consumption Overplot Tool
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
REBCO_PER_ARC_KM = 5730.0

# ============================================================
# USER INPUT
# ============================================================

MODE = "BE_FIXED"   # "CAGR_FIXED" or "BE_FIXED"

CAGR_FIXED = 4
BE_LIST = [1, 2, 4, 6, 8, 10]

BE_FIXED = 2
CAGR_LIST = [1, 2, 4, 6, 8, 10]

# ============================================================
# Helper: fleet power
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
    build_times = build_times / 12.0

    n_arc = np.arange(1, len(build_times) + 1)
    fleet_power = n_arc * ARC_POWER_MWE / 1e3

    return build_times, fleet_power


# ============================================================
# Helper: Li consumption (cumulative)
# ============================================================

def get_li_curve(be, cagr):

    sqlite_file = f"RUNS/ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF/out_ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF.sqlite"

    if not os.path.exists(sqlite_file):
        return None, None

    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    cur.execute("""
    SELECT T.Time, R.Quantity
    FROM Transactions T
    JOIN Resources R
        ON T.ResourceId = R.ResourceId
    WHERE
        T.Commodity = 'Li_raw'
        AND T.ReceiverId IN (
            SELECT AgentId FROM AgentEntry
            WHERE Prototype = 'FLiBeFab'
        )
    ORDER BY T.Time
    """)

    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        return None, None

    times = np.array([r[0] for r in rows], dtype=float)
    qty   = np.array([r[1] for r in rows], dtype=float)

    times = times / 12.0
    cum_li = np.cumsum(qty) / 1e3   # kg → ton

    return times, cum_li


# ============================================================
# Helper: REBCO demand (from build events)
# ============================================================

def get_rebco_curve(be, cagr):

    t, _ = get_fleet_curve(be, cagr)

    if t is None:
        return None, None

    n_arc = np.arange(1, len(t) + 1)
    rebco = n_arc * REBCO_PER_ARC_KM

    return t, rebco


# ============================================================
# Plot: Fleet Power
# ============================================================

def plot_power():

    plt.figure()
    colors = plt.cm.tab10.colors

    if MODE == "CAGR_FIXED":
        for idx, be in enumerate(BE_LIST):
            t, p = get_fleet_curve(be, CAGR_FIXED)
            if t is None:
                continue

            plt.step(t, p, where='post',
                     color=colors[idx % len(colors)],
                     label=f"Be = {be} t/mo")

        plt.title(f"Deployment (CAGR = {CAGR_FIXED}%)")

    elif MODE == "BE_FIXED":
        for idx, cagr in enumerate(CAGR_LIST):
            t, p = get_fleet_curve(BE_FIXED, cagr)
            if t is None:
                continue

            plt.step(t, p, where='post',
                     color=colors[idx % len(colors)],
                     label=f"CAGR = {cagr}%")

        plt.title(f"Deployment (Be = {BE_FIXED} t/mo)")

    plt.xlabel("Time (year)")
    plt.ylabel("Fleet Power (GWe)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot: Li consumption
# ============================================================

def plot_li():

    plt.figure()
    colors = plt.cm.tab10.colors

    if MODE == "CAGR_FIXED":
        for idx, be in enumerate(BE_LIST):
            t, li = get_li_curve(be, CAGR_FIXED)
            if t is None:
                continue

            plt.step(t, li/1e3, where='post',
                     color=colors[idx % len(colors)],
                     label=f"Be = {be} t/mo")

        plt.title(f"Li Consumption (CAGR = {CAGR_FIXED}%)")

    elif MODE == "BE_FIXED":
        for idx, cagr in enumerate(CAGR_LIST):
            t, li = get_li_curve(BE_FIXED, cagr)
            if t is None:
                continue

            plt.step(t, li/1e3, where='post',
                     color=colors[idx % len(colors)],
                     label=f"CAGR = {cagr}%")

        plt.title(f"Li Consumption (Be = {BE_FIXED} t/mo)")

    plt.xlabel("Time (year)")
    plt.ylabel("Cumulative Li (k-ton)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot: REBCO demand
# ============================================================

def get_rebco_supply_curve(t_max_years, growth=0.10, p0=2400.0):

    years = np.arange(0, int(np.ceil(t_max_years)) + 1)

    annual_prod = p0 * (1 + growth) ** years
    cumulative = np.cumsum(annual_prod)

    return years, cumulative

def plot_rebco():

    plt.figure()
    colors = plt.cm.tab10.colors

    t_max = 0.0  # supply curve용

    if MODE == "CAGR_FIXED":
        for idx, be in enumerate(BE_LIST):
            t, r = get_rebco_curve(be, CAGR_FIXED)
            if t is None:
                continue

            t_max = max(t_max, np.max(t))

            plt.step(t, r, where='post',
                     color=colors[idx % len(colors)],
                     label=f"Be = {be} t/mo")

        plt.title(f"REBCO Demand (CAGR = {CAGR_FIXED}%)")

    elif MODE == "BE_FIXED":
        for idx, cagr in enumerate(CAGR_LIST):
            t, r = get_rebco_curve(BE_FIXED, cagr)
            if t is None:
                continue

            t_max = max(t_max, np.max(t))

            plt.step(t, r, where='post',
                     color=colors[idx % len(colors)],
                     label=f"CAGR = {cagr}%")

        plt.title(f"REBCO Demand (Be = {BE_FIXED} t/mo)")

    # ========================================================
    # ADD: REBCO SUPPLY CURVE (12% CAGR)
    # ========================================================

    t_supply, s_supply = get_rebco_supply_curve(t_max,0.10)

    plt.plot(t_supply, s_supply,
             linestyle='--',
             linewidth=2.5,
             color='black',
             label="REBCO CAGR: 10%")

    # ========================================================

    plt.xlabel("Time (year)")
    plt.ylabel("Cumulative REBCO (km)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    plot_power()
    plot_li()
    plot_rebco()