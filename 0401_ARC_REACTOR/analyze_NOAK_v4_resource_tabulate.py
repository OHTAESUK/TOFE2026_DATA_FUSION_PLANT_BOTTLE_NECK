#!/usr/bin/env python3
# ============================================================
# Export ARC monthly resource requirements from NOAK sweep
# ============================================================

import os
import sqlite3
import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

RUN_DIR = "RUNS"
OUT_CSV = "NOAK_ARC_resource_requirements_monthly.csv"

BE_RANGE = range(1, 11)       # t/month
CAGR_RANGE = range(0, 11)     # %

DURATION_MONTHS = 600

ARC_POWER_MWE = 190.0
REBCO_PER_ARC_KM = 5730.0

LI90_STARTUP_T_PER_ARC = 132.6
LI6_BURN_KG_PER_ARC_YR = 57.8
LI90_ENRICHMENT = 0.90

LI90_OP_T_PER_ARC_YR = LI6_BURN_KG_PER_ARC_YR / LI90_ENRICHMENT / 1000.0
LI90_OP_T_PER_ARC_MONTH = LI90_OP_T_PER_ARC_YR / 12.0


def read_build_times(be, cagr):
    tag = f"ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF"
    sqlite_file = os.path.join(RUN_DIR, tag, f"out_{tag}.sqlite")

    if not os.path.exists(sqlite_file):
        return None

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

    if not rows:
        return np.array([], dtype=int)

    return np.array([r[0] for r in rows], dtype=int)


def process_case(be, cagr):
    build_times = read_build_times(be, cagr)
    if build_times is None:
        return []

    rows = []

    li90_operation_cum_t = 0.0

    for month in range(1, DURATION_MONTHS + 1):

        n_built = int(np.sum(build_times <= month))
        n_operating = n_built  # NO RETIREMENT IN CURRENT ANALYSIS
        
        li90_startup_cum_t = n_built * LI90_STARTUP_T_PER_ARC

        li90_operation_cum_t += n_operating * LI90_OP_T_PER_ARC_MONTH

        li90_total_cum_t = li90_startup_cum_t + li90_operation_cum_t

        rebco_cum_km = n_built * REBCO_PER_ARC_KM

        fleet_power_gwe = n_built * ARC_POWER_MWE / 1000.0

        rows.append({
            "case": f"Be{be}_CAGR{cagr}",
            "be_t_per_month": be,
            "cagr_percent": cagr,
            "time_month": month,
            "time_year": month / 12.0,
            "n_arc_built": n_built,
            "n_arc_operating": n_operating,
            "fleet_power_gwe": fleet_power_gwe,
            "li90_startup_cum_t": li90_startup_cum_t,
            "li90_operation_cum_t": li90_operation_cum_t,
            "li90_total_cum_t": li90_total_cum_t,
            "rebco_cum_km": rebco_cum_km,
        })

    return rows


def main():
    all_rows = []

    for be in BE_RANGE:
        for cagr in CAGR_RANGE:
            print(f"[INFO] Processing Be={be} t/mo, CAGR={cagr}%")
            all_rows.extend(process_case(be, cagr))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    print("--------------------------------------------------")
    print(f"[INFO] Wrote: {OUT_CSV}")
    print(f"[INFO] Rows : {len(df)}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()