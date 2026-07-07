#!/usr/bin/env python3
# ============================================================
# Compact Tokamak REBCO / Be Cumulative Demand vs Supply
# FAST / SLOW deployment scenarios
# Be supply cases can vary either initial supply or CAGR
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# USER SETTINGS
# ============================================================

START_YEAR = 2050
END_YEAR = 2100
BASE_SUPPLY_YEAR = 2024

SCENARIOS_TO_PLOT = ["FAST", "SLOW"]

# Be replacement intervals to plot
BE_REPLACEMENT_INTERVALS_YR = [5, 10]

# Supply assumptions
#
# Be supply mode:
#   "CAGR_FIXED" : keep Be CAGR fixed and vary the initial 2024 Be supply
#   "BE_FIXED"   : keep the initial 2024 Be supply fixed and vary Be CAGR
BE_SUPPLY_MODE = "BE_FIXED"   # "CAGR_FIXED" or "BE_FIXED"

# Case 1: fixed CAGR, varying initial Be supply
BE_SUPPLY_CAGR_FIXED = 0.04
BE_SUPPLY_2024_T_PER_MONTH_LIST = [1, 2, 4, 6, 8]

# Case 2: fixed initial Be supply, varying CAGR
BE_SUPPLY_2024_T_PER_MONTH_FIXED = 8.0
BE_SUPPLY_CAGR_LIST = [0.00, 0.02, 0.04, 0.06, 0.08]

REBCO_SUPPLY_2024_KM_PER_YR = 2400.0
REBCO_SUPPLY_CAGR = 0.10

Y_AXIS_PAD_FRACTION = 0.08

# ============================================================
# Deployment targets
# ============================================================

TABLE5_TARGETS_GWE = {
    "FAST": {
        2050: 3.0,
        2060: 11.0,
        2070: 38.0,
        2080: 152.0,
        2090: 395.0,
        2100: 1024.0,
    },
    "SLOW": {
        2050: 1.0,
        2060: 4.0,
        2070: 14.0,
        2080: 27.0,
        2090: 71.0,
        2100: 184.0,
    },
}

# ============================================================
# Compact Tokamak parameters
# ============================================================

CT_NET_ELECTRIC_MWE = 190.0
CT_REBCO_KM_PER_PLANT = 5730.0

CT_BE_STARTUP_T_PER_PLANT = 88.7
CT_BE_REPLACEMENT_T_PER_PLANT = 3.82

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "legend.frameon": False,
})

SCENARIO_COLORS = {
    "FAST": "tab:blue",
    "SLOW": "tab:green",
}

SCENARIO_LINESTYLES = {
    "FAST": "-",
    "SLOW": "--",
}

# ============================================================
# Utility functions
# ============================================================

def human_format(x, pos):
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def nice_upper_limit(values, pad_fraction=0.08):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return 1.0

    vmax = float(np.max(finite))
    if vmax <= 0.0:
        return 1.0

    padded = vmax * (1.0 + pad_fraction)
    exponent = math.floor(math.log10(padded))
    base = 10 ** exponent
    fraction = padded / base

    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 1.2:
        nice_fraction = 1.2
    elif fraction <= 1.5:
        nice_fraction = 1.5
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 3.0:
        nice_fraction = 3.0
    elif fraction <= 4.0:
        nice_fraction = 4.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    elif fraction <= 7.5:
        nice_fraction = 7.5
    else:
        nice_fraction = 10.0

    return nice_fraction * base


def set_axis_style(ax, ylabel, title, ylim_max=None):
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))

    if ylim_max is not None:
        ax.set_ylim(0.0, ylim_max)


def interpolate_capacity_targets(target_dict):
    years_known = np.array(sorted(target_dict.keys()), dtype=float)
    values_known = np.array([target_dict[int(y)] for y in years_known], dtype=float)

    years = np.arange(START_YEAR, END_YEAR + 1, dtype=int)
    values = np.interp(years, years_known, values_known)

    return pd.DataFrame({
        "year": years,
        "target_capacity_gwe": values,
    })


def required_ct_plants(capacity_gwe):
    ct_net_gwe = CT_NET_ELECTRIC_MWE / 1000.0
    return int(math.ceil(capacity_gwe / ct_net_gwe))


def get_rebco_supply_curve():
    """
    REBCO supply curve independent of demand.

    Cumulative supply is counted from BASE_SUPPLY_YEAR to each year:
        sum_y REBCO_SUPPLY_2024_KM_PER_YR * (1 + REBCO_SUPPLY_CAGR)^(y - BASE_SUPPLY_YEAR)
    The returned curve is clipped to START_YEAR-END_YEAR only for plotting.
    """
    years_full = np.arange(BASE_SUPPLY_YEAR, END_YEAR + 1, dtype=int)

    annual_supply_km = (
        REBCO_SUPPLY_2024_KM_PER_YR
        * (1.0 + REBCO_SUPPLY_CAGR) ** (years_full - BASE_SUPPLY_YEAR)
    )
    cumulative_supply_km = np.cumsum(annual_supply_km)

    df = pd.DataFrame({
        "year": years_full,
        "annual_supply_km": annual_supply_km,
        "cumulative_supply_km": cumulative_supply_km,
    })

    return df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()


def get_be_supply_curve(initial_t_per_month, cagr):
    """
    Be supply curve independent of demand.

    Cumulative supply is counted from BASE_SUPPLY_YEAR to each year:
        sum_y (initial_t_per_month * 12) * (1 + cagr)^(y - BASE_SUPPLY_YEAR)
    The returned curve is clipped to START_YEAR-END_YEAR only for plotting.
    """
    years_full = np.arange(BASE_SUPPLY_YEAR, END_YEAR + 1, dtype=int)

    annual_supply_t = (
        initial_t_per_month * 12.0
        * (1.0 + cagr) ** (years_full - BASE_SUPPLY_YEAR)
    )
    cumulative_supply_t = np.cumsum(annual_supply_t)

    df = pd.DataFrame({
        "year": years_full,
        "annual_supply_t": annual_supply_t,
        "cumulative_supply_t": cumulative_supply_t,
    })

    return df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()


def get_be_supply_cases():
    """Return Be supply cases according to BE_SUPPLY_MODE."""
    cases = []

    if BE_SUPPLY_MODE == "CAGR_FIXED":
        for initial in BE_SUPPLY_2024_T_PER_MONTH_LIST:
            cases.append({
                "initial_t_per_month": float(initial),
                "cagr": float(BE_SUPPLY_CAGR_FIXED),
                "label": f"Supply: {initial:g} t/mo, {BE_SUPPLY_CAGR_FIXED * 100:.0f}% CAGR",
            })

    elif BE_SUPPLY_MODE == "BE_FIXED":
        for cagr in BE_SUPPLY_CAGR_LIST:
            cases.append({
                "initial_t_per_month": float(BE_SUPPLY_2024_T_PER_MONTH_FIXED),
                "cagr": float(cagr),
                "label": f"Supply: {BE_SUPPLY_2024_T_PER_MONTH_FIXED:g} t/mo, {cagr * 100:.0f}% CAGR",
            })

    else:
        raise ValueError("Invalid BE_SUPPLY_MODE. Use 'CAGR_FIXED' or 'BE_FIXED'.")

    return cases


# ============================================================
# Demand calculation
# ============================================================

def build_ct_scenario(scenario_name):
    df = interpolate_capacity_targets(TABLE5_TARGETS_GWE[scenario_name])

    required_plants = []
    new_builds = []

    previous_required = None

    for _, row in df.iterrows():
        n_required = required_ct_plants(row["target_capacity_gwe"])

        # 2050 target fleet is included as actual startup demand.
        if previous_required is None:
            n_new = n_required
        else:
            n_new = max(n_required - previous_required, 0)

        required_plants.append(n_required)
        new_builds.append(n_new)
        previous_required = n_required

    df["scenario"] = scenario_name
    df["required_plants"] = required_plants
    df["new_builds"] = new_builds

    df["rebco_annual_demand_km"] = df["new_builds"] * CT_REBCO_KM_PER_PLANT
    df["rebco_cumulative_demand_km"] = df["rebco_annual_demand_km"].cumsum()

    return df


def add_be_demand(df, replacement_interval_yr):
    out = df.copy()

    out["be_startup_annual_demand_t"] = (
        out["new_builds"] * CT_BE_STARTUP_T_PER_PLANT
    )

    replacement_demand = []

    for year in out["year"]:
        demand_this_year = 0.0

        for _, cohort in out.iterrows():
            build_year = int(cohort["year"])
            n_built = int(cohort["new_builds"])

            if n_built <= 0:
                continue

            years_since_build = int(year - build_year)

            if years_since_build <= 0:
                continue

            if years_since_build % replacement_interval_yr == 0:
                demand_this_year += n_built * CT_BE_REPLACEMENT_T_PER_PLANT

        replacement_demand.append(demand_this_year)

    out["be_replacement_interval_yr"] = replacement_interval_yr
    out["be_replacement_annual_demand_t"] = replacement_demand
    out["be_total_annual_demand_t"] = (
        out["be_startup_annual_demand_t"]
        + out["be_replacement_annual_demand_t"]
    )
    out["be_cumulative_demand_t"] = out["be_total_annual_demand_t"].cumsum()

    return out


def build_all_ct_results():
    base_frames = []

    for scenario_name in SCENARIOS_TO_PLOT:
        base_frames.append(build_ct_scenario(scenario_name))

    base_df = pd.concat(base_frames, ignore_index=True)

    be_results = []

    for interval in BE_REPLACEMENT_INTERVALS_YR:
        frames = []
        for scenario_name in SCENARIOS_TO_PLOT:
            sub = base_df[base_df["scenario"] == scenario_name].copy()
            frames.append(add_be_demand(sub, interval))

        be_results.append((interval, pd.concat(frames, ignore_index=True)))

    return base_df, be_results


# ============================================================
# Plot: REBCO cumulative demand vs supply
# ============================================================

def plot_rebco_cumulative_demand_vs_supply(base_df):
    supply_df = get_rebco_supply_curve()

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_values = list(supply_df["cumulative_supply_km"].values)

    for scenario_name in SCENARIOS_TO_PLOT:
        sub = base_df[base_df["scenario"] == scenario_name].copy()
        all_values.extend(sub["rebco_cumulative_demand_km"].values)

        ax.plot(
            sub["year"],
            sub["rebco_cumulative_demand_km"],
            color=SCENARIO_COLORS[scenario_name],
            linestyle=SCENARIO_LINESTYLES[scenario_name],
            label=f"{scenario_name} demand",
        )

    ax.plot(
        supply_df["year"],
        supply_df["cumulative_supply_km"],
        color="black",
        linestyle="--",
        linewidth=2.8,
        label=(
            f"Supply: {BASE_SUPPLY_YEAR} "
            f"{REBCO_SUPPLY_2024_KM_PER_YR:,.0f} km/yr, "
            f"{REBCO_SUPPLY_CAGR * 100:.0f}% CAGR"
        ),
    )

    set_axis_style(
        ax=ax,
        ylabel="Cumulative REBCO (km)",
        title="Compact Tokamak Cumulative REBCO Demand vs Supply",
        ylim_max=nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION),
    )

    fig.tight_layout()
    plt.show()


# ============================================================
# Plot: Be cumulative demand vs supply
# ============================================================

def plot_be_cumulative_demand_vs_supply(be_results):
    be_supply_cases = get_be_supply_cases()
    supply_colors = plt.cm.Greys(np.linspace(0.35, 0.80, len(be_supply_cases)))
    supply_linestyles = ["--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]

    for interval, df in be_results:
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        all_values = []

        for scenario_name in SCENARIOS_TO_PLOT:
            sub = df[df["scenario"] == scenario_name].copy()
            all_values.extend(sub["be_cumulative_demand_t"].values)

            ax.plot(
                sub["year"],
                sub["be_cumulative_demand_t"],
                color=SCENARIO_COLORS[scenario_name],
                linestyle=SCENARIO_LINESTYLES[scenario_name],
                linewidth=2.8,
                label=f"{scenario_name} demand",
            )

        for idx, case in enumerate(be_supply_cases):
            supply_df = get_be_supply_curve(
                initial_t_per_month=case["initial_t_per_month"],
                cagr=case["cagr"],
            )
            all_values.extend(supply_df["cumulative_supply_t"].values)

            ax.plot(
                supply_df["year"],
                supply_df["cumulative_supply_t"],
                color=supply_colors[idx],
                linestyle=supply_linestyles[idx % len(supply_linestyles)],
                linewidth=2.4,
                label=case["label"],
            )

        set_axis_style(
            ax=ax,
            ylabel="Cumulative Be (t)",
            title=f"Compact Tokamak Cumulative Be Demand vs Supply ({interval}-yr replacement)",
            ylim_max=nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION),
        )

        fig.tight_layout()
        plt.show()


# ============================================================
# Console summary
# ============================================================

def print_summary(base_df, be_results):
    rebco_supply = get_rebco_supply_curve()
    be_supply_cases = get_be_supply_cases()

    print("")
    print("============================================================")
    print("Compact Tokamak REBCO / Be cumulative demand vs supply")
    print("============================================================")
    print(f"Deployment years              : {START_YEAR}-{END_YEAR}")
    print(f"Supply base year              : {BASE_SUPPLY_YEAR}")
    print(f"REBCO supply in 2024          : {REBCO_SUPPLY_2024_KM_PER_YR:,.0f} km/yr")
    print(f"REBCO supply CAGR             : {REBCO_SUPPLY_CAGR * 100:.1f}%")
    print(f"Be supply mode                : {BE_SUPPLY_MODE}")
    print("------------------------------------------------------------")
    print(f"Cumulative REBCO supply by {END_YEAR}: {rebco_supply.iloc[-1]['cumulative_supply_km']:,.0f} km")
    print("------------------------------------------------------------")

    for case in be_supply_cases:
        be_supply = get_be_supply_curve(
            initial_t_per_month=case["initial_t_per_month"],
            cagr=case["cagr"],
        )
        print(
            f"Cumulative Be supply by {END_YEAR}   : "
            f"{be_supply.iloc[-1]['cumulative_supply_t']:,.1f} t "
            f"({case['label']})"
        )

    print("============================================================")
    print("")

    for scenario_name in SCENARIOS_TO_PLOT:
        sub = base_df[base_df["scenario"] == scenario_name]
        last = sub.iloc[-1]

        print("------------------------------------------------------------")
        print(f"[{scenario_name}] Compact Tokamak")
        print(f"Final target capacity         : {last['target_capacity_gwe']:.1f} GWe")
        print(f"Required plants in {END_YEAR} : {int(last['required_plants'])}")
        print(f"Cumulative REBCO demand       : {last['rebco_cumulative_demand_km']:,.0f} km")
        print("------------------------------------------------------------")

    for interval, df in be_results:
        for scenario_name in SCENARIOS_TO_PLOT:
            sub = df[df["scenario"] == scenario_name]
            last = sub.iloc[-1]

            print("------------------------------------------------------------")
            print(f"[{scenario_name}] Be, {interval}-yr replacement")
            print(f"Cumulative Be demand          : {last['be_cumulative_demand_t']:,.1f} t")
            print(f"Final-year annual Be demand   : {last['be_total_annual_demand_t']:,.1f} t")
            print("------------------------------------------------------------")


# ============================================================
# MAIN
# ============================================================

def main():
    base_df, be_results = build_all_ct_results()

    print_summary(base_df, be_results)

    plot_rebco_cumulative_demand_vs_supply(base_df)
    plot_be_cumulative_demand_vs_supply(be_results)


if __name__ == "__main__":
    main()