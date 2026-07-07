#!/usr/bin/env python3
# ============================================================
# Representative fusion reactor concept deployment comparison
# Compact Tokamak + Tandem Mirror + Large-scale Tokamak
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = "NOAK_ARC_resource_requirements_monthly.csv"

BE_FIXED = 2          # t/month
CAGR_FIXED = 4        # %

SAVE_CSV = False
OUT_CSV = f"Combined_from_ARC_Li_Be{BE_FIXED}_CAGR{CAGR_FIXED}.csv"

# ============================================================
# DISPLAY NAMES
# ============================================================

NAME_ARC = "Compact Tokamak"
NAME_HAMMIR = "Tandem Mirror"
NAME_EUDEMO = "Large-scale Tokamak"

# ============================================================
# ARC PARAMETERS
# ============================================================

ARC_FUSION_POWER_MW = 525.0
ARC_NET_ELECTRIC_MWE = 190.0
ARC_REBCO_KM_PER_PLANT = 5730.0

# ============================================================
# Hammir PARAMETERS / PbLi Volume Doubled
# ============================================================

HAMMIR_FUSION_POWER_MW = 350.0
HAMMIR_NET_ELECTRIC_MWE = 165.0
HAMMIR_REBCO_KM_PER_PLANT = 1406.0

HAMMIR_LI30_STARTUP_T_PER_PLANT = 20.1 * 2
HAMMIR_LI6_BURN_KG_PER_YR = 38.5
HAMMIR_LI30_ENRICHMENT = 0.30
HAMMIR_PBLI_T_PER_PLANT = 3352.1 * 2
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
    "font.size": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.3,
    "legend.frameon": False,
})

# ============================================================
# Fixed colors for consistency across figures
# ============================================================

COLOR_ARC = "tab:blue"
COLOR_HAMMIR = "tab:orange"
COLOR_EUDEMO = "tab:green"

# ============================================================
# LOAD ARC RESOURCE CSV
# ============================================================

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Cannot find input CSV: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

case = df[
    (df["be_t_per_month"] == BE_FIXED)
    & (df["cagr_percent"] == CAGR_FIXED)
].copy()

if case.empty:
    raise ValueError(
        f"No matching case found for Be={BE_FIXED}, CAGR={CAGR_FIXED}"
    )

case = case.sort_values("time_month").reset_index(drop=True)

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
# Axis formatter
# ============================================================

def human_format(x, pos):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    else:
        return f"{x:.0f}"

# ============================================================
# Monthly deployment bookkeeping
# ============================================================

records = []

# Hammir inventory and state
h_inventory_li30_t = 0.0
h_n_built = 0
h_li30_startup_cum_t = 0.0
h_li30_operation_cum_t = 0.0

# EU-DEMO inventory and state
e_inventory_li90_t = 0.0
e_n_built = 0
e_li90_startup_cum_t = 0.0
e_li90_operation_cum_t = 0.0

for idx, row in case.iterrows():

    month = int(row["time_month"])
    year = float(row["time_year"])

    # ARC quantities
    arc_n_built = int(row["n_arc_built"])
    arc_fusion_power_gw = arc_n_built * ARC_FUSION_POWER_MW / 1000.0
    arc_net_electric_gwe = arc_n_built * ARC_NET_ELECTRIC_MWE / 1000.0
    arc_rebco_km = arc_n_built * ARC_REBCO_KM_PER_PLANT

    li90_supply_monthly_t = float(row["li90_supply_monthly_t"])
    li90_supply_cum_t = float(row["li90_supply_cum_t"])

    li30_supply_monthly_t = float(row["li30_supply_monthly_t"])
    li30_supply_cum_t = float(row["li30_supply_cum_t"])

    # Hammir deployment using Li30
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

    # EU-DEMO deployment using Li90
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

out = pd.DataFrame(records)

if SAVE_CSV:
    out.to_csv(OUT_CSV, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================

last = out.iloc[-1]

print("--------------------------------------------------")
print(f"[INFO] Case: Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%")
print(f"[INFO] Li90 -> Li30 factor for Tandem Mirror: {LI90_TO_LI30_FACTOR:.4f}")
print("--------------------------------------------------")
print(f"[{NAME_ARC}] Plants:        {last['arc_n_built']:.0f}")
print(f"[{NAME_ARC}] Fusion power:  {last['arc_fusion_power_gw']:.2f} GW")
print(f"[{NAME_ARC}] Net electric:  {last['arc_net_electric_gwe']:.2f} GWe")
print(f"[{NAME_ARC}] REBCO:         {last['arc_rebco_km']:,.0f} km")
print(f"[{NAME_ARC}] Li90 supply:   {last['arc_li90_supply_cum_t']:,.2f} t")
print("--------------------------------------------------")
print(f"[{NAME_HAMMIR}] Plants:        {last['hammir_n_built']:.0f}")
print(f"[{NAME_HAMMIR}] Fusion power:  {last['hammir_fusion_power_gw']:.2f} GW")
print(f"[{NAME_HAMMIR}] Net electric:  {last['hammir_net_electric_gwe']:.2f} GWe")
print(f"[{NAME_HAMMIR}] REBCO:         {last['hammir_rebco_km']:,.0f} km")
print(f"[{NAME_HAMMIR}] PbLi:          {last['hammir_pbli_t']:,.0f} t")
print(f"[{NAME_HAMMIR}] WB:            {last['hammir_wb_t']:,.0f} t")
print(f"[{NAME_HAMMIR}] Li30 used:     {last['hammir_li30_total_used_t']:,.2f} t")
print(f"[{NAME_HAMMIR}] Li30 remain:   {last['hammir_li30_inventory_remaining_t']:,.2f} t")
print("--------------------------------------------------")
print(f"[{NAME_EUDEMO}] Plants:        {last['eudemo_n_built']:.0f}")
print(f"[{NAME_EUDEMO}] Fusion power:  {last['eudemo_fusion_power_gw']:.2f} GW")
print(f"[{NAME_EUDEMO}] Net electric:  {last['eudemo_net_electric_gwe']:.2f} GWe")
print(f"[{NAME_EUDEMO}] REBCO:         {last['eudemo_rebco_km']:,.0f} km")
print(f"[{NAME_EUDEMO}] PbLi:          {last['eudemo_pbli_t']:,.0f} t")
print(f"[{NAME_EUDEMO}] Nb3Sn:         {last['eudemo_nb3sn_t']:,.0f} t")
print(f"[{NAME_EUDEMO}] Li90 used:     {last['eudemo_li90_total_used_t']:,.2f} t")
print(f"[{NAME_EUDEMO}] Li90 remain:   {last['eudemo_li90_inventory_remaining_t']:,.2f} t")
print("--------------------------------------------------")
if SAVE_CSV:
    print(f"[INFO] Wrote: {OUT_CSV}")
print("[INFO] Combined analysis complete.")
print("--------------------------------------------------")

# ============================================================
# PLOTS
# ============================================================

t = out["time_year"]

# 1. Plant count
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_n_built"], where="post", color=COLOR_ARC, linestyle="-", label=NAME_ARC)
plt.step(t, out["hammir_n_built"], where="post", color=COLOR_HAMMIR, linestyle="-", label=NAME_HAMMIR)
plt.step(t, out["eudemo_n_built"], where="post", color=COLOR_EUDEMO, linestyle="--", label=NAME_EUDEMO)
plt.xlabel("Time (year)")
plt.ylabel("Number of Plants")
plt.title(f"Deployment Count (Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(FuncFormatter(human_format))
plt.show()

# 2. Fusion power
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_fusion_power_gw"], where="post", color=COLOR_ARC, linestyle="-", label=NAME_ARC)
plt.step(t, out["hammir_fusion_power_gw"], where="post", color=COLOR_HAMMIR, linestyle="-", label=NAME_HAMMIR)
plt.step(t, out["eudemo_fusion_power_gw"], where="post", color=COLOR_EUDEMO, linestyle="-", label=NAME_EUDEMO)
plt.xlabel("Time (year)")
plt.ylabel("Fusion Power (GW)")
plt.title("Fusion Power Deployment")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(FuncFormatter(human_format))
plt.show()

# 3. Net electric power
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_net_electric_gwe"], where="post", color=COLOR_ARC, label=NAME_ARC)
plt.step(t, out["hammir_net_electric_gwe"], where="post", color=COLOR_HAMMIR, label=NAME_HAMMIR)
plt.step(t, out["eudemo_net_electric_gwe"], where="post", color=COLOR_EUDEMO, label=NAME_EUDEMO)
plt.xlabel("Time (year)")
plt.ylabel("Net Electric Power (GWe)")
plt.title("Net Electric Power Deployment")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(FuncFormatter(human_format))
plt.show()

# 6. REBCO comparison
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_rebco_km"], where="post", color=COLOR_ARC, label=NAME_ARC)
plt.step(t, out["hammir_rebco_km"], where="post", color=COLOR_HAMMIR, label=NAME_HAMMIR)
plt.step(t, out["eudemo_rebco_km"], where="post", color=COLOR_EUDEMO, label=NAME_EUDEMO)
plt.xlabel("Time (year)")
plt.ylabel("Cumulative REBCO (km)")
plt.title("REBCO Requirement")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(FuncFormatter(human_format))
plt.show()

# 7. PbLi comparison
plt.figure(figsize=(8, 5))
plt.step(t, out["hammir_pbli_t"], where="post", color=COLOR_HAMMIR, label=NAME_HAMMIR)
plt.step(t, out["eudemo_pbli_t"], where="post", color=COLOR_EUDEMO, label=NAME_EUDEMO)
plt.xlabel("Time (year)")
plt.ylabel("Cumulative PbLi Requirement (t)")
plt.title("PbLi Requirement")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(FuncFormatter(human_format))
plt.show()

# ============================================================
# 8-9. WB and Nb3Sn Requirement (Dual Y-axis)
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis: WB
ax1.step(
    t,
    out["hammir_wb_t"],
    where="post",
    color=COLOR_HAMMIR,
    linestyle="--",
    label=f"{NAME_HAMMIR} WB",
)

ax1.set_xlabel("Time (year)")
ax1.set_ylabel(
    "Cumulative WB Requirement (t)",
    color=COLOR_HAMMIR,
)
ax1.tick_params(axis="y", labelcolor=COLOR_HAMMIR)
ax1.yaxis.set_major_formatter(FuncFormatter(human_format))
ax1.grid(True, alpha=0.3)

# Right axis: Nb3Sn
ax2 = ax1.twinx()

ax2.step(
    t,
    out["eudemo_nb3sn_t"],
    where="post",
    color=COLOR_EUDEMO,
    linestyle="-",
    label=rf"{NAME_EUDEMO} Nb$_3$Sn",
)

ax2.set_ylabel(
    r"Cumulative Nb$_3$Sn Requirement (t)",
    color=COLOR_EUDEMO,
)
ax2.tick_params(axis="y", labelcolor=COLOR_EUDEMO)
ax2.yaxis.set_major_formatter(FuncFormatter(human_format))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
)

plt.title(r"Tungsten Boride (WB) and Nb$_3$Sn Requirements")
plt.tight_layout()
plt.show()