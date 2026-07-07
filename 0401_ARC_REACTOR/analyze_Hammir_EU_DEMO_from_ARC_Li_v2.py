#!/usr/bin/env python3
# ============================================================
# Hammir + EU-DEMO WCLL deployment from ARC-derived Li supply
# 2D Be-CAGR ratio maps at a user-specified target year
#
# Input:
#   NOAK_ARC_resource_requirements_monthly.csv
#
# Output:
#   Ratio heatmap/contour plots
#   Optional summary CSV
# ============================================================

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = "NOAK_ARC_resource_requirements_monthly.csv"

# Plot ratios at this elapsed time.
# Example: TARGET_YEAR = 30.0 means ratios after 30 years.
TARGET_YEAR = 50.0
TARGET_MONTH = int(round(TARGET_YEAR * 12))

# If None, all Be/CAGR values available in the CSV are used.
# Otherwise, use lists such as [1, 2, 3, 4, 5].
BE_VALUES = None
CAGR_VALUES = None

SAVE_CSV = False
OUT_CSV = f"Be_CAGR_ratio_summary_year{TARGET_YEAR:g}.csv"

SHOW_CONTOURS = False
N_CONTOUR_LEVELS = 7


# ============================================================
# ARC PARAMETERS
# ============================================================

ARC_FUSION_POWER_MW = 525.0
ARC_NET_ELECTRIC_MWE = 190.0
ARC_REBCO_KM_PER_PLANT = 5730.0


# ============================================================
# Hammir PARAMETERS
# ============================================================

HAMMIR_FUSION_POWER_MW = 350.0
HAMMIR_NET_ELECTRIC_MWE = 165.0
HAMMIR_REBCO_KM_PER_PLANT = 1406.0

HAMMIR_LI30_STARTUP_T_PER_PLANT = 20.1*2
HAMMIR_LI6_BURN_KG_PER_YR = 38.5
HAMMIR_LI30_ENRICHMENT = 0.30
HAMMIR_PBLI_T_PER_PLANT = 3352.1*2
HAMMIR_WB_T_PER_PLANT = 781.18

HAMMIR_LI30_OP_T_PER_PLANT_MONTH = (
    HAMMIR_LI6_BURN_KG_PER_YR / HAMMIR_LI30_ENRICHMENT / 1000.0 / 12.0
)


# ============================================================
# EU-DEMO WCLL PARAMETERS
# ============================================================

EUDEMO_FUSION_POWER_MW = 5000.0
EUDEMO_NET_ELECTRIC_MWE = 1550.0

EUDEMO_LI90_STARTUP_T_PER_PLANT = 133.4
EUDEMO_LI6_BURN_KG_PER_YR = 550.0
EUDEMO_LI90_ENRICHMENT = 0.90

EUDEMO_PBLI_T_PER_PLANT = 24371.0
# EUDEMO_NB3SN_T_PER_PLANT = 1285.0
# EUDEMO_REBCO_KM_PER_PLANT = 1542.0

EUDEMO_NB3SN_T_PER_PLANT = 595.0
EUDEMO_REBCO_KM_PER_PLANT = 714.0

EUDEMO_LI90_OP_T_PER_PLANT_MONTH = (
    EUDEMO_LI6_BURN_KG_PER_YR / EUDEMO_LI90_ENRICHMENT / 1000.0 / 12.0
)


# ============================================================
# Li blending: ARC Li90 -> Hammir Li30
# ============================================================

LI90_ENRICHMENT = 0.90
LI30_TARGET_ENRICHMENT = 0.30
LI_NAT_ENRICHMENT = 0.075

LI90_TO_LI30_FACTOR = (
    (LI90_ENRICHMENT - LI_NAT_ENRICHMENT)
    / (LI30_TARGET_ENRICHMENT - LI_NAT_ENRICHMENT)
)


# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.3,
    "legend.frameon": False,
})


# ============================================================
# Utility functions
# ============================================================

def safe_ratio(numerator, denominator):
    if abs(denominator) < 1e-15:
        return np.nan
    return numerator / denominator


def nice_step(raw_step):
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0

    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0

    return nice_fraction * (10 ** exponent)


def nice_limits(values, n_levels=7):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return 0.0, 1.0, np.linspace(0.0, 1.0, n_levels)

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))

    if abs(vmax - vmin) < 1e-12:
        center = 0.0 if abs(vmin) < 1e-12 else vmin
        span = max(abs(center) * 0.2, 1.0)
        raw_min = center - span / 2.0
        raw_max = center + span / 2.0
    else:
        span = vmax - vmin
        pad = 0.05 * span
        raw_min = vmin - pad
        raw_max = vmax + pad

    raw_step = (raw_max - raw_min) / max(n_levels - 1, 1)
    step = nice_step(raw_step)

    nice_min = math.floor(raw_min / step) * step
    nice_max = math.ceil(raw_max / step) * step

    if nice_min == nice_max:
        nice_min -= step
        nice_max += step

    levels = np.arange(nice_min, nice_max + 0.5 * step, step)

    if levels.size < 3:
        levels = np.linspace(nice_min, nice_max, n_levels)

    return nice_min, nice_max, levels


def print_progress_header(total_cases):
    print("")
    print("============================================================")
    print("Starting Be-CAGR ratio-map calculation")
    print(f"Target year : {TARGET_YEAR:.1f} yr")
    print(f"Target month: {TARGET_MONTH}")
    print(f"Total cases : {total_cases}")
    print("============================================================")
    print("")


def print_progress(case_counter, total_cases, be, cagr, start_time):
    elapsed = time.time() - start_time
    avg_time = elapsed / max(case_counter, 1)
    eta = avg_time * max(total_cases - case_counter, 0)

    print(
        f"[{case_counter:>4d}/{total_cases:<4d}] "
        f"Be={float(be):>7.2f} t/mo | "
        f"CAGR={float(cagr):>6.2f}% | "
        f"elapsed={elapsed:>8.1f}s | "
        f"ETA={eta:>8.1f}s",
        flush=True,
    )


def print_progress_footer(n_processed, elapsed_total):
    print("")
    print("============================================================")
    print("Calculation completed successfully")
    print(f"Processed cases: {n_processed}")
    print(f"Elapsed time   : {elapsed_total:.1f} s")
    print("============================================================")
    print("")


# ============================================================
# Future feasibility checks
# ============================================================

def can_build_one_more_hammir(
    inventory_now_t,
    current_count,
    current_month_index,
    future_li30_monthly_supply,
):
    test_inventory = inventory_now_t - HAMMIR_LI30_STARTUP_T_PER_PLANT
    if test_inventory < -1e-9:
        return False

    test_count = current_count + 1

    for future_month in range(current_month_index + 1, len(future_li30_monthly_supply)):
        test_inventory += future_li30_monthly_supply[future_month]
        test_inventory -= test_count * HAMMIR_LI30_OP_T_PER_PLANT_MONTH

        if test_inventory < -1e-9:
            return False

    return True


def can_build_one_more_eudemo(
    inventory_now_t,
    current_count,
    current_month_index,
    future_li90_monthly_supply,
):
    test_inventory = inventory_now_t - EUDEMO_LI90_STARTUP_T_PER_PLANT
    if test_inventory < -1e-9:
        return False

    test_count = current_count + 1

    for future_month in range(current_month_index + 1, len(future_li90_monthly_supply)):
        test_inventory += future_li90_monthly_supply[future_month]
        test_inventory -= test_count * EUDEMO_LI90_OP_T_PER_PLANT_MONTH

        if test_inventory < -1e-9:
            return False

    return True


# ============================================================
# Single-case deployment calculation
# ============================================================

def run_one_case(case):
    case = case.sort_values("time_month").reset_index(drop=True).copy()

    case["li90_supply_monthly_t"] = (
        case["li90_total_cum_t"].diff().fillna(case["li90_total_cum_t"])
    )

    case["li90_supply_cum_t"] = case["li90_supply_monthly_t"].cumsum()

    case["li30_supply_monthly_t"] = (
        case["li90_supply_monthly_t"] * LI90_TO_LI30_FACTOR
    )

    case["li30_supply_cum_t"] = case["li30_supply_monthly_t"].cumsum()

    future_li90_monthly_supply = case["li90_supply_monthly_t"].values
    future_li30_monthly_supply = case["li30_supply_monthly_t"].values

    records = []

    h_inventory_li30_t = 0.0
    h_n_built = 0
    h_li30_startup_cum_t = 0.0
    h_li30_operation_cum_t = 0.0

    e_inventory_li90_t = 0.0
    e_n_built = 0
    e_li90_startup_cum_t = 0.0
    e_li90_operation_cum_t = 0.0

    for idx, row in case.iterrows():
        month = int(row["time_month"])
        year = float(row["time_year"])

        arc_n_built = int(row["n_arc_built"])
        arc_fusion_power_gw = arc_n_built * ARC_FUSION_POWER_MW / 1000.0
        arc_net_electric_gwe = arc_n_built * ARC_NET_ELECTRIC_MWE / 1000.0
        arc_rebco_km = arc_n_built * ARC_REBCO_KM_PER_PLANT

        li90_supply_monthly_t = float(row["li90_supply_monthly_t"])
        li90_supply_cum_t = float(row["li90_supply_cum_t"])

        li30_supply_monthly_t = float(row["li30_supply_monthly_t"])
        li30_supply_cum_t = float(row["li30_supply_cum_t"])

        h_inventory_li30_t += li30_supply_monthly_t

        h_op_this_month_t = h_n_built * HAMMIR_LI30_OP_T_PER_PLANT_MONTH
        h_inventory_li30_t -= h_op_this_month_t
        h_li30_operation_cum_t += h_op_this_month_t

        if h_inventory_li30_t < 0 and abs(h_inventory_li30_t) < 1e-9:
            h_inventory_li30_t = 0.0

        if h_inventory_li30_t < -1e-9:
            raise RuntimeError(
                f"Hammir Li30 inventory became negative at month {month}: "
                f"{h_inventory_li30_t:.6f} t"
            )

        h_new_builds = 0

        while h_inventory_li30_t >= HAMMIR_LI30_STARTUP_T_PER_PLANT:
            feasible = can_build_one_more_hammir(
                inventory_now_t=h_inventory_li30_t,
                current_count=h_n_built + h_new_builds,
                current_month_index=idx,
                future_li30_monthly_supply=future_li30_monthly_supply,
            )

            if not feasible:
                break

            h_inventory_li30_t -= HAMMIR_LI30_STARTUP_T_PER_PLANT
            h_new_builds += 1

        h_startup_this_month_t = h_new_builds * HAMMIR_LI30_STARTUP_T_PER_PLANT
        h_li30_startup_cum_t += h_startup_this_month_t
        h_n_built += h_new_builds

        h_li30_total_used_t = h_li30_startup_cum_t + h_li30_operation_cum_t

        h_fusion_power_gw = h_n_built * HAMMIR_FUSION_POWER_MW / 1000.0
        h_net_electric_gwe = h_n_built * HAMMIR_NET_ELECTRIC_MWE / 1000.0
        h_rebco_km = h_n_built * HAMMIR_REBCO_KM_PER_PLANT
        h_pbli_t = h_n_built * HAMMIR_PBLI_T_PER_PLANT
        h_wb_t = h_n_built * HAMMIR_WB_T_PER_PLANT

        e_inventory_li90_t += li90_supply_monthly_t

        e_op_this_month_t = e_n_built * EUDEMO_LI90_OP_T_PER_PLANT_MONTH
        e_inventory_li90_t -= e_op_this_month_t
        e_li90_operation_cum_t += e_op_this_month_t

        if e_inventory_li90_t < 0 and abs(e_inventory_li90_t) < 1e-9:
            e_inventory_li90_t = 0.0

        if e_inventory_li90_t < -1e-9:
            raise RuntimeError(
                f"EU-DEMO Li90 inventory became negative at month {month}: "
                f"{e_inventory_li90_t:.6f} t"
            )

        e_new_builds = 0

        while e_inventory_li90_t >= EUDEMO_LI90_STARTUP_T_PER_PLANT:
            feasible = can_build_one_more_eudemo(
                inventory_now_t=e_inventory_li90_t,
                current_count=e_n_built + e_new_builds,
                current_month_index=idx,
                future_li90_monthly_supply=future_li90_monthly_supply,
            )

            if not feasible:
                break

            e_inventory_li90_t -= EUDEMO_LI90_STARTUP_T_PER_PLANT
            e_new_builds += 1

        e_startup_this_month_t = e_new_builds * EUDEMO_LI90_STARTUP_T_PER_PLANT
        e_li90_startup_cum_t += e_startup_this_month_t
        e_n_built += e_new_builds

        e_li90_total_used_t = e_li90_startup_cum_t + e_li90_operation_cum_t

        e_fusion_power_gw = e_n_built * EUDEMO_FUSION_POWER_MW / 1000.0
        e_net_electric_gwe = e_n_built * EUDEMO_NET_ELECTRIC_MWE / 1000.0
        e_rebco_km = e_n_built * EUDEMO_REBCO_KM_PER_PLANT
        e_pbli_t = e_n_built * EUDEMO_PBLI_T_PER_PLANT
        e_nb3sn_t = e_n_built * EUDEMO_NB3SN_T_PER_PLANT

        records.append({
            "time_month": month,
            "time_year": year,

            "arc_n_built": arc_n_built,
            "arc_fusion_power_gw": arc_fusion_power_gw,
            "arc_net_electric_gwe": arc_net_electric_gwe,
            "arc_rebco_km": arc_rebco_km,
            "arc_li90_supply_cum_t": li90_supply_cum_t,

            "hammir_new_builds": h_new_builds,
            "hammir_n_built": h_n_built,
            "hammir_fusion_power_gw": h_fusion_power_gw,
            "hammir_net_electric_gwe": h_net_electric_gwe,
            "hammir_rebco_km": h_rebco_km,
            "hammir_pbli_t": h_pbli_t,
            "hammir_wb_t": h_wb_t,
            "hammir_li30_total_used_t": h_li30_total_used_t,
            "hammir_li30_inventory_remaining_t": h_inventory_li30_t,
            "li30_supply_cum_t": li30_supply_cum_t,

            "eudemo_new_builds": e_new_builds,
            "eudemo_n_built": e_n_built,
            "eudemo_fusion_power_gw": e_fusion_power_gw,
            "eudemo_net_electric_gwe": e_net_electric_gwe,
            "eudemo_rebco_km": e_rebco_km,
            "eudemo_pbli_t": e_pbli_t,
            "eudemo_nb3sn_t": e_nb3sn_t,
            "eudemo_li90_total_used_t": e_li90_total_used_t,
            "eudemo_li90_inventory_remaining_t": e_inventory_li90_t,
        })

    return pd.DataFrame(records)


def summarize_one_case(be_value, cagr_value, case):
    out = run_one_case(case)

    eligible = out[out["time_month"] <= TARGET_MONTH]

    if eligible.empty:
        row = out.iloc[0]
    else:
        row = eligible.iloc[-1]

    return {
        "be_t_per_month": be_value,
        "cagr_percent": cagr_value,
        "target_year": TARGET_YEAR,
        "actual_time_month": int(row["time_month"]),
        "actual_time_year": float(row["time_year"]),

        "arc_n_built": int(row["arc_n_built"]),
        "hammir_n_built": int(row["hammir_n_built"]),
        "eudemo_n_built": int(row["eudemo_n_built"]),

        "arc_fusion_power_gw": float(row["arc_fusion_power_gw"]),
        "hammir_fusion_power_gw": float(row["hammir_fusion_power_gw"]),
        "eudemo_fusion_power_gw": float(row["eudemo_fusion_power_gw"]),

        "arc_net_electric_gwe": float(row["arc_net_electric_gwe"]),
        "hammir_net_electric_gwe": float(row["hammir_net_electric_gwe"]),
        "eudemo_net_electric_gwe": float(row["eudemo_net_electric_gwe"]),

        "hammir_to_arc_fusion_ratio": safe_ratio(
            row["hammir_fusion_power_gw"],
            row["arc_fusion_power_gw"],
        ),
        "eudemo_to_arc_fusion_ratio": safe_ratio(
            row["eudemo_fusion_power_gw"],
            row["arc_fusion_power_gw"],
        ),
        "hammir_to_arc_electric_ratio": safe_ratio(
            row["hammir_net_electric_gwe"],
            row["arc_net_electric_gwe"],
        ),
        "eudemo_to_arc_electric_ratio": safe_ratio(
            row["eudemo_net_electric_gwe"],
            row["arc_net_electric_gwe"],
        ),
    }


# ============================================================
# Plotting
# ============================================================

def plot_ratio_map(summary, value_col, title, cbar_label):
    pivot = summary.pivot(
        index="cagr_percent",
        columns="be_t_per_month",
        values=value_col,
    ).sort_index().sort_index(axis=1)

    x = pivot.columns.values.astype(float)
    y = pivot.index.values.astype(float)
    z = pivot.values.astype(float)

    zmin, zmax, levels = nice_limits(z, n_levels=N_CONTOUR_LEVELS)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    mesh = ax.pcolormesh(
        x,
        y,
        z,
        shading="auto",
        vmin=zmin,
        vmax=zmax,
    )
    
    # ============================================================
    # Add cell values
    # ============================================================
    
    x_centers = (x[:-1] + x[1:]) / 2 if len(x) > 1 else x
    y_centers = (y[:-1] + y[1:]) / 2 if len(y) > 1 else y
    
    # If pcolormesh grid aligns directly with x/y values
    if len(x_centers) != z.shape[1]:
        x_centers = x
    
    if len(y_centers) != z.shape[0]:
        y_centers = y
    
    for iy, y_val in enumerate(y_centers):
        for ix, x_val in enumerate(x_centers):
    
            value = z[iy, ix]
    
            if np.isfinite(value):
                ax.text(
                    x_val,
                    y_val,
                    f"{value:.1f}",
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if SHOW_CONTOURS:
        finite_z = z[np.isfinite(z)]

        if finite_z.size > 0 and np.nanmax(finite_z) > np.nanmin(finite_z):
            cs = ax.contour(
                x,
                y,
                z,
                levels=levels,
                colors="black",
                linewidths=0.8,
                alpha=0.75,
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")

    ax.set_xlabel("Initial Be supply (t/month)")
    ax.set_ylabel("Be supply CAGR (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()


# ============================================================
# Main driver
# ============================================================

def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Cannot find input CSV: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    required_cols = {
        "be_t_per_month",
        "cagr_percent",
        "time_month",
        "time_year",
        "n_arc_built",
        "li90_total_cum_t",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    be_values = (
        sorted(df["be_t_per_month"].unique())
        if BE_VALUES is None
        else BE_VALUES
    )

    cagr_values = (
        sorted(df["cagr_percent"].unique())
        if CAGR_VALUES is None
        else CAGR_VALUES
    )

    total_cases = len(be_values) * len(cagr_values)
    case_counter = 0
    start_time = time.time()

    print_progress_header(total_cases)

    summary_records = []

    for be in be_values:
        for cagr in cagr_values:
            case_counter += 1
            print_progress(case_counter, total_cases, be, cagr, start_time)

            case = df[
                (df["be_t_per_month"] == be)
                & (df["cagr_percent"] == cagr)
            ].copy()

            if case.empty:
                print("    skipped: no matching case", flush=True)
                continue

            summary_records.append(
                summarize_one_case(
                    be_value=be,
                    cagr_value=cagr,
                    case=case,
                )
            )

    elapsed_total = time.time() - start_time
    print_progress_footer(len(summary_records), elapsed_total)

    summary = pd.DataFrame(summary_records)

    if summary.empty:
        raise RuntimeError("No Be-CAGR cases were processed.")

    if SAVE_CSV:
        summary.to_csv(OUT_CSV, index=False)
        print(f"[INFO] Wrote: {OUT_CSV}")

    time_label = f"Year {TARGET_YEAR:g}"

    plot_ratio_map(
        summary,
        "hammir_to_arc_fusion_ratio",
        f"{time_label} Fusion Power Ratio: Hammir / ARC",
        "Hammir / ARC fusion power",
    )

    plot_ratio_map(
        summary,
        "eudemo_to_arc_fusion_ratio",
        f"{time_label} Fusion Power Ratio: EU-DEMO / ARC",
        "EU-DEMO / ARC fusion power",
    )

    plot_ratio_map(
        summary,
        "hammir_to_arc_electric_ratio",
        f"{time_label} Net Electric Power Ratio: Hammir / ARC",
        "Hammir / ARC net electric power",
    )

    plot_ratio_map(
        summary,
        "eudemo_to_arc_electric_ratio",
        f"{time_label} Net Electric Power Ratio: EU-DEMO / ARC",
        "EU-DEMO / ARC net electric power",
    )


if __name__ == "__main__":
    main()
