#!/usr/bin/env python3
# ============================================================
# EU-DEMO WCLL deployment from ARC-derived Li90 supply
# v1: single Be-CAGR case comparison
#
# - ARC deployment/resource schedule is read from:
#   NOAK_ARC_resource_requirements_monthly.csv
# - EU-DEMO WCLL deployment is evaluated using the same
#   time-dependent Li90 supply schedule.
# - New EU-DEMO units are accepted only if future operational
#   Li90 demand can be sustained without negative inventory.
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

OUT_CSV = f"EU_DEMO_from_ARC_Li_Be{BE_FIXED}_CAGR{CAGR_FIXED}.csv"

# ============================================================
# ARC PARAMETERS
# ============================================================

ARC_FUSION_POWER_MW = 525.0
ARC_NET_ELECTRIC_MWE = 190.0
ARC_REBCO_KM_PER_PLANT = 5730.0

# ============================================================
# EU-DEMO WCLL PARAMETERS
# ============================================================

EUDEMO_FUSION_POWER_MW = 5000.0
EUDEMO_NET_ELECTRIC_MWE = 1550.0

EUDEMO_LI90_STARTUP_T_PER_PLANT = 133.4
EUDEMO_LI6_BURN_KG_PER_YR = 550.0
EUDEMO_LI90_ENRICHMENT = 0.90

# EUDEMO_REBCO_KM_PER_PLANT = 1542.0
# EUDEMO_NB3SN_T_PER_PLANT = 1285.0
EUDEMO_PBLI_T_PER_PLANT = 24371.0

EUDEMO_REBCO_KM_PER_PLANT = 714.0
EUDEMO_NB3SN_T_PER_PLANT = 595.0

EUDEMO_LI90_OP_T_PER_PLANT_YR = (
    EUDEMO_LI6_BURN_KG_PER_YR / EUDEMO_LI90_ENRICHMENT / 1000.0
)

EUDEMO_LI90_OP_T_PER_PLANT_MONTH = (
    EUDEMO_LI90_OP_T_PER_PLANT_YR / 12.0
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
# Precompute monthly Li90 supply from ARC-derived Li90
# ============================================================

case["li90_supply_monthly_t"] = (
    case["li90_total_cum_t"].diff().fillna(case["li90_total_cum_t"])
)

case["li90_supply_cum_t"] = (
    case["li90_supply_monthly_t"].cumsum()
)

future_li90_monthly_supply = case["li90_supply_monthly_t"].values

# ============================================================
# Future feasibility check
# ============================================================

def can_build_one_more(
    inventory_now_t,
    current_eudemo_count,
    current_month_index,
    future_li90_monthly_supply,
):
    """
    Check whether adding one more EU-DEMO unit now keeps Li90 inventory
    non-negative for all future months.

    Startup Li is paid immediately.
    Future operational Li demand is checked over the remaining simulation.
    """

    test_inventory = (
        inventory_now_t - EUDEMO_LI90_STARTUP_T_PER_PLANT
    )

    if test_inventory < -1e-9:
        return False

    test_eudemo_count = current_eudemo_count + 1

    for future_month in range(
        current_month_index + 1,
        len(future_li90_monthly_supply)
    ):
        test_inventory += future_li90_monthly_supply[future_month]

        test_inventory -= (
            test_eudemo_count
            * EUDEMO_LI90_OP_T_PER_PLANT_MONTH
        )

        if test_inventory < -1e-9:
            return False

    return True

# ============================================================
# EU-DEMO monthly deployment bookkeeping
# ============================================================

records = []

inventory_li90_t = 0.0

eudemo_n_built = 0
eudemo_li90_startup_cum_t = 0.0
eudemo_li90_operation_cum_t = 0.0

for idx, row in case.iterrows():

    month = int(row["time_month"])
    year = float(row["time_year"])

    li90_supply_cum_t = float(row["li90_supply_cum_t"])
    li90_supply_monthly_t = float(row["li90_supply_monthly_t"])

    # --------------------------------------------------------
    # 1. Add this month's Li90 supply
    # --------------------------------------------------------

    inventory_li90_t += li90_supply_monthly_t

    # --------------------------------------------------------
    # 2. Existing EU-DEMO fleet consumes operational Li90
    # --------------------------------------------------------

    li90_op_this_month_t = (
        eudemo_n_built * EUDEMO_LI90_OP_T_PER_PLANT_MONTH
    )

    inventory_li90_t -= li90_op_this_month_t
    eudemo_li90_operation_cum_t += li90_op_this_month_t

    if inventory_li90_t < 0 and abs(inventory_li90_t) < 1e-9:
        inventory_li90_t = 0.0

    if inventory_li90_t < -1e-9:
        raise RuntimeError(
            f"Li90 inventory became negative at month {month}: "
            f"{inventory_li90_t:.6f} t"
        )

    # --------------------------------------------------------
    # 3. Future-sustainable deployment
    # --------------------------------------------------------

    eudemo_new_builds = 0

    while inventory_li90_t >= EUDEMO_LI90_STARTUP_T_PER_PLANT:

        feasible = can_build_one_more(
            inventory_now_t=inventory_li90_t,
            current_eudemo_count=eudemo_n_built + eudemo_new_builds,
            current_month_index=idx,
            future_li90_monthly_supply=future_li90_monthly_supply,
        )

        if not feasible:
            break

        inventory_li90_t -= EUDEMO_LI90_STARTUP_T_PER_PLANT
        eudemo_new_builds += 1

    startup_used_this_month_t = (
        eudemo_new_builds * EUDEMO_LI90_STARTUP_T_PER_PLANT
    )

    eudemo_li90_startup_cum_t += startup_used_this_month_t
    eudemo_n_built += eudemo_new_builds

    eudemo_li90_total_used_t = (
        eudemo_li90_startup_cum_t + eudemo_li90_operation_cum_t
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
    # EU-DEMO quantities
    # --------------------------------------------------------

    eudemo_fusion_power_gw = (
        eudemo_n_built * EUDEMO_FUSION_POWER_MW / 1000.0
    )

    eudemo_net_electric_gwe = (
        eudemo_n_built * EUDEMO_NET_ELECTRIC_MWE / 1000.0
    )

    eudemo_rebco_km = (
        eudemo_n_built * EUDEMO_REBCO_KM_PER_PLANT
    )

    eudemo_pbli_t = (
        eudemo_n_built * EUDEMO_PBLI_T_PER_PLANT
    )

    eudemo_nb3sn_t = (
        eudemo_n_built * EUDEMO_NB3SN_T_PER_PLANT
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

        # EU-DEMO
        "eudemo_new_builds": eudemo_new_builds,
        "eudemo_n_built": eudemo_n_built,
        "eudemo_fusion_power_gw": eudemo_fusion_power_gw,
        "eudemo_net_electric_gwe": eudemo_net_electric_gwe,
        "eudemo_rebco_km": eudemo_rebco_km,
        "eudemo_pbli_t": eudemo_pbli_t,
        "eudemo_nb3sn_t": eudemo_nb3sn_t,
        "eudemo_li90_startup_cum_t": eudemo_li90_startup_cum_t,
        "eudemo_li90_operation_cum_t": eudemo_li90_operation_cum_t,
        "eudemo_li90_total_used_t": eudemo_li90_total_used_t,
        "eudemo_li90_inventory_remaining_t": inventory_li90_t,
    })

out = pd.DataFrame(records)
# out.to_csv(OUT_CSV, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================

last = out.iloc[-1]

print("--------------------------------------------------")
print(f"[INFO] Case: Be={BE_FIXED} t/mo, CAGR={CAGR_FIXED}%")
print("--------------------------------------------------")
print(f"[ARC]     Plants: {last['arc_n_built']:.0f}")
print(f"[ARC]     Fusion power: {last['arc_fusion_power_gw']:.2f} GW")
print(f"[ARC]     Net electric:  {last['arc_net_electric_gwe']:.2f} GWe")
print(f"[ARC]     REBCO:         {last['arc_rebco_km']:,.0f} km")
print(f"[ARC]     Li90 supply:   {last['arc_li90_supply_cum_t']:,.2f} t")
print("--------------------------------------------------")
print(f"[EU-DEMO] Plants: {last['eudemo_n_built']:.0f}")
print(f"[EU-DEMO] Fusion power: {last['eudemo_fusion_power_gw']:.2f} GW")
print(f"[EU-DEMO] Net electric:  {last['eudemo_net_electric_gwe']:.2f} GWe")
print(f"[EU-DEMO] REBCO:         {last['eudemo_rebco_km']:,.0f} km")
print(f"[EU-DEMO] PbLi:          {last['eudemo_pbli_t']:,.0f} t")
print(f"[EU-DEMO] Nb3Sn:         {last['eudemo_nb3sn_t']:,.0f} t")
print(f"[EU-DEMO] Li90 used:     {last['eudemo_li90_total_used_t']:,.2f} t")
print(f"[EU-DEMO] Li90 remain:   {last['eudemo_li90_inventory_remaining_t']:,.2f} t")
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
plt.step(t, out["eudemo_n_built"], where="post", label="EU-DEMO WCLL")
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
plt.step(t, out["eudemo_fusion_power_gw"], where="post", label="EU-DEMO WCLL")
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
plt.step(t, out["eudemo_net_electric_gwe"], where="post", label="EU-DEMO WCLL")
plt.xlabel("Time (year)")
plt.ylabel("Net Electric Power (GWe)")
plt.title("Net Electric Power Deployment")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Li90 supply and EU-DEMO use
plt.figure(figsize=(8, 5))
plt.plot(t, out["arc_li90_supply_cum_t"], label="Li-90 supply from ARC case")
plt.plot(t, out["eudemo_li90_total_used_t"], label="EU-DEMO Li-90 used")
plt.plot(
    t,
    out["eudemo_li90_inventory_remaining_t"],
    linestyle="--",
    label="Remaining Li-90 inventory",
)
plt.xlabel("Time (year)")
plt.ylabel("Li-90 Equivalent (t)")
plt.title("Li-90 Supply and EU-DEMO Use")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 5. REBCO comparison
plt.figure(figsize=(8, 5))
plt.step(t, out["arc_rebco_km"], where="post", label="ARC")
plt.step(t, out["eudemo_rebco_km"], where="post", label="EU-DEMO WCLL")
plt.xlabel("Time (year)")
plt.ylabel("Cumulative REBCO (km)")
plt.title("REBCO Requirement")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 6. EU-DEMO PbLi requirement
plt.figure(figsize=(8, 5))
plt.step(t, out["eudemo_pbli_t"], where="post", label="EU-DEMO PbLi")
plt.xlabel("Time (year)")
plt.ylabel("Cumulative PbLi Requirement (t)")
plt.title("EU-DEMO WCLL PbLi Requirement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. EU-DEMO Nb3Sn requirement
plt.figure(figsize=(8, 5))
plt.step(t, out["eudemo_nb3sn_t"], where="post", label="EU-DEMO Nb3Sn")
plt.xlabel("Time (year)")
plt.ylabel("Cumulative Nb3Sn Requirement (t)")
plt.title("EU-DEMO WCLL Nb3Sn Requirement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("[INFO] EU-DEMO-from-ARC-Li v1 analysis complete.")