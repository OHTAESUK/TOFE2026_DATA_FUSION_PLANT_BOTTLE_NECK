#!/usr/bin/env python3
# ============================================================
# Hammir deployment from ARC-derived Li supply
# v1: single Be-CAGR case comparison
#
# Revision:
# - Future-feasible greedy deployment
# - A new Hammir unit is deployed only if startup + future
#   operational Li demand can be sustained without negative inventory.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = "NOAK_ARC_resource_requirements_monthly.csv"

BE_FIXED = 2          # t/month
CAGR_FIXED = 4        # %

OUT_CSV = f"Hammir_from_ARC_Li_Be{BE_FIXED}_CAGR{CAGR_FIXED}.csv"

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

HAMMIR_LI30_STARTUP_T_PER_PLANT = 20.1
HAMMIR_LI6_BURN_KG_PER_YR = 38.5
HAMMIR_LI30_ENRICHMENT = 0.30

HAMMIR_PBLI_T_PER_PLANT = 3352.1
HAMMIR_WB_T_PER_PLANT = 781.18

HAMMIR_LI30_OP_T_PER_PLANT_YR = (
    HAMMIR_LI6_BURN_KG_PER_YR / HAMMIR_LI30_ENRICHMENT / 1000.0
)

HAMMIR_LI30_OP_T_PER_PLANT_MONTH = (
    HAMMIR_LI30_OP_T_PER_PLANT_YR / 12.0
)

# ============================================================
# Li blending: 90% enriched Li -> 30% enriched Li
# using natural Li as diluent
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

# ============================================================
# Precompute monthly Li30 supply from ARC-derived Li90
# ============================================================

case["li90_supply_monthly_t"] = (
    case["li90_total_cum_t"].diff().fillna(case["li90_total_cum_t"])
)

case["li30_supply_monthly_t"] = (
    case["li90_supply_monthly_t"] * LI90_TO_LI30_FACTOR
)

case["li30_supply_cum_t"] = (
    case["li30_supply_monthly_t"].cumsum()
)

future_li30_monthly_supply = case["li30_supply_monthly_t"].values

# ============================================================
# Future feasibility check
# ============================================================

def can_build_one_more(
    inventory_now_t,
    current_hammir_count,
    current_month_index,
    future_li30_monthly_supply,
):
    """
    Check whether adding one more Hammir plant now keeps Li30 inventory
    non-negative for all future months.

    The function assumes:
    - startup Li for the candidate plant is paid immediately;
    - the candidate plant joins the operating fleet from the next
      future bookkeeping steps used in this feasibility check.
    """

    test_inventory = (
        inventory_now_t - HAMMIR_LI30_STARTUP_T_PER_PLANT
    )

    if test_inventory < -1e-9:
        return False

    test_hammir_count = current_hammir_count + 1

    # Check all future months after the current one.
    # The current month operation has already been charged in the main loop.
    for future_month in range(
        current_month_index + 1,
        len(future_li30_monthly_supply)
    ):

        test_inventory += future_li30_monthly_supply[future_month]

        test_inventory -= (
            test_hammir_count
            * HAMMIR_LI30_OP_T_PER_PLANT_MONTH
        )

        if test_inventory < -1e-9:
            return False

    return True

# ============================================================
# Hammir monthly deployment bookkeeping
# ============================================================

records = []

inventory_li30_t = 0.0

hammir_n_built = 0
hammir_li30_startup_cum_t = 0.0
hammir_li30_operation_cum_t = 0.0

for idx, row in case.iterrows():

    month = int(row["time_month"])
    year = float(row["time_year"])

    li90_supply_cum_t = float(row["li90_total_cum_t"])
    li90_supply_monthly_t = float(row["li90_supply_monthly_t"])

    li30_supply_monthly_t = float(row["li30_supply_monthly_t"])
    li30_supply_cum_t = float(row["li30_supply_cum_t"])

    # --------------------------------------------------------
    # 1. Add this month's Li30 supply
    # --------------------------------------------------------

    inventory_li30_t += li30_supply_monthly_t

    # --------------------------------------------------------
    # 2. Existing Hammir fleet consumes operational Li
    # --------------------------------------------------------

    li30_op_this_month_t = (
        hammir_n_built * HAMMIR_LI30_OP_T_PER_PLANT_MONTH
    )

    inventory_li30_t -= li30_op_this_month_t
    hammir_li30_operation_cum_t += li30_op_this_month_t

    if inventory_li30_t < 0 and abs(inventory_li30_t) < 1e-9:
        inventory_li30_t = 0.0

    # If this happens, previous deployment was infeasible.
    # This should not occur with the future-feasible deployment rule.
    if inventory_li30_t < -1e-9:
        raise RuntimeError(
            f"Li30 inventory became negative at month {month}: "
            f"{inventory_li30_t:.6f} t"
        )

    # --------------------------------------------------------
    # 3. Future-feasible greedy deployment
    # --------------------------------------------------------

    hammir_new_builds = 0

    while inventory_li30_t >= HAMMIR_LI30_STARTUP_T_PER_PLANT:

        feasible = can_build_one_more(
            inventory_now_t=inventory_li30_t,
            current_hammir_count=hammir_n_built + hammir_new_builds,
            current_month_index=idx,
            future_li30_monthly_supply=future_li30_monthly_supply,
        )

        if not feasible:
            break

        inventory_li30_t -= HAMMIR_LI30_STARTUP_T_PER_PLANT
        hammir_new_builds += 1

    startup_used_this_month_t = (
        hammir_new_builds * HAMMIR_LI30_STARTUP_T_PER_PLANT
    )

    hammir_li30_startup_cum_t += startup_used_this_month_t
    hammir_n_built += hammir_new_builds

    hammir_li30_total_used_t = (
        hammir_li30_startup_cum_t + hammir_li30_operation_cum_t
    )

    # --------------------------------------------------------
    # ARC quantities
    # --------------------------------------------------------

    arc_n_built = int(row["n_arc_built"])

    arc_fusion_power_gw = (
        arc_n_built * ARC_FUSION_POWER_MW / 1000.0
    )

    arc_net_electric_gwe = (
        arc_n_built * ARC_NET_ELECTRIC_MWE / 1000.0
    )

    arc_rebco_km = (
        arc_n_built * ARC_REBCO_KM_PER_PLANT
    )

    # --------------------------------------------------------
    # Hammir quantities
    # --------------------------------------------------------

    hammir_fusion_power_gw = (
        hammir_n_built * HAMMIR_FUSION_POWER_MW / 1000.0
    )

    hammir_net_electric_gwe = (
        hammir_n_built * HAMMIR_NET_ELECTRIC_MWE / 1000.0
    )

    hammir_rebco_km = (
        hammir_n_built * HAMMIR_REBCO_KM_PER_PLANT
    )

    hammir_pbli_t = (
        hammir_n_built * HAMMIR_PBLI_T_PER_PLANT
    )

    hammir_wb_t = (
        hammir_n_built * HAMMIR_WB_T_PER_PLANT
    )

    records.append({
        "be_t_per_month": BE_FIXED,
        "cagr_percent": CAGR_FIXED,
        "time_month": month,
        "time_year": year,

        # ARC
        "arc_n_built": arc_n_built,
        "arc_fusion_power_gw": arc_fusion_power_gw,
        "arc_net_electric_gwe": arc_net_electric_gwe,
        "arc_li90_supply_cum_t": li90_supply_cum_t,
        "arc_li90_supply_monthly_t": li90_supply_monthly_t,
        "arc_rebco_km": arc_rebco_km,

        # Li conversion
        "li90_to_li30_factor": LI90_TO_LI30_FACTOR,
        "li30_supply_monthly_t": li30_supply_monthly_t,
        "li30_supply_cum_t": li30_supply_cum_t,

        # Hammir
        "hammir_new_builds": hammir_new_builds,
        "hammir_n_built": hammir_n_built,
        "hammir_fusion_power_gw": hammir_fusion_power_gw,
        "hammir_net_electric_gwe": hammir_net_electric_gwe,
        "hammir_rebco_km": hammir_rebco_km,
        "hammir_pbli_t": hammir_pbli_t,
        "hammir_wb_t": hammir_wb_t,
        "hammir_li30_startup_cum_t": hammir_li30_startup_cum_t,
        "hammir_li30_operation_cum_t": hammir_li30_operation_cum_t,
        "hammir_li30_total_used_t": hammir_li30_total_used_t,
        "hammir_li30_inventory_remaining_t": inventory_li30_t,
    })

out = pd.DataFrame(records)
# out.to_csv(OUT_CSV, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================

last = out.iloc[-1]

print("--------------------------------------------------")
print(f"[INFO] Case: Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%")
print(f"[INFO] Li90 -> Li30 factor: {LI90_TO_LI30_FACTOR:.4f}")
print("--------------------------------------------------")
print(f"[ARC]    Plants: {last['arc_n_built']:.0f}")
print(f"[ARC]    Fusion power: {last['arc_fusion_power_gw']:.2f} GW")
print(f"[ARC]    Net electric:  {last['arc_net_electric_gwe']:.2f} GWe")
print(f"[ARC]    REBCO:         {last['arc_rebco_km']:,.0f} km")
print(f"[ARC]    Li90 supply:   {last['arc_li90_supply_cum_t']:,.2f} t")
print("--------------------------------------------------")
print(f"[Hammir] Plants: {last['hammir_n_built']:.0f}")
print(f"[Hammir] Fusion power: {last['hammir_fusion_power_gw']:.2f} GW")
print(f"[Hammir] Net electric:  {last['hammir_net_electric_gwe']:.2f} GWe")
print(f"[Hammir] REBCO:         {last['hammir_rebco_km']:,.0f} km")
print(f"[Hammir] PbLi:          {last['hammir_pbli_t']:,.0f} t")
print(f"[Hammir] WB:            {last['hammir_wb_t']:,.0f} t")
print(f"[Hammir] Li30 used:     {last['hammir_li30_total_used_t']:,.2f} t")
print(f"[Hammir] Li30 remain:   {last['hammir_li30_inventory_remaining_t']:,.2f} t")
print("--------------------------------------------------")
# print(f"[INFO] Wrote: {OUT_CSV}")
print("--------------------------------------------------")

# ============================================================
# PLOTS
# ============================================================

t = out["time_year"]

# 1. Plant count
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_n_built"], where="post", label="ARC")
plt.step(t, out["hammir_n_built"], where="post", label="Hammir")
plt.xlabel("Time (year)")
plt.ylabel("Number of Plants")
plt.title(f"Deployment Count (Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Fusion power
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_fusion_power_gw"], where="post", label="ARC")
plt.step(t, out["hammir_fusion_power_gw"], where="post", label="Hammir")
plt.xlabel("Time (year)")
plt.ylabel("Fusion Power (GW)")
plt.title("Fusion Power Deployment")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Net electric power
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_net_electric_gwe"], where="post", label="ARC")
plt.step(t, out["hammir_net_electric_gwe"], where="post", label="Hammir")
plt.xlabel("Time (year)")
plt.ylabel("Net Electric Power (GWe)")
plt.title("Net Electric Power Deployment")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Li-30 supply and Hammir use
plt.figure(figsize=(8, 5))
plt.plot(t, out["li30_supply_cum_t"], label="Li-30 supply from ARC Li90")
plt.plot(t, out["hammir_li30_total_used_t"], label="Hammir Li-30 used")
plt.plot(
    t,
    out["hammir_li30_inventory_remaining_t"],
    linestyle="--",
    label="Remaining Li-30 inventory",
)
plt.xlabel("Time (year)")
plt.ylabel("Li-30 Equivalent (t)")
plt.title("Li-30 Supply and Hammir Use")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 5. REBCO comparison
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_rebco_km"], where="post", label="ARC")
plt.step(t, out["hammir_rebco_km"], where="post", label="Hammir")
plt.xlabel("Time (year)")
plt.ylabel("Cumulative REBCO (km)")
plt.title("REBCO Requirement")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 6. Hammir PbLi requirement
plt.figure(figsize=(8, 5))
plt.step(t, out["hammir_pbli_t"], where="post", label="Hammir PbLi")
plt.xlabel("Time (year)")
plt.ylabel("Cumulative PbLi Requirement (t)")
plt.title("Hammir PbLi Requirement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Hammir WB requirement
plt.figure(figsize=(8, 5))
plt.step(t, out["hammir_wb_t"], where="post", label="Hammir WB")
plt.xlabel("Time (year)")
plt.ylabel("Cumulative WB Requirement (t)")
plt.title("Hammir Tungsten Boride (WB) Requirement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("[INFO] Hammir-from-ARC-Li v1 analysis complete.")