#!/usr/bin/env python3
# ============================================================
# demand_response_Fusion_Reactors_v1.py
#
# ARC material annual demand from Table 5 fusion deployment scenarios
#
# Purpose:
#   Estimate annual ARC material demand required to satisfy the
#   FAST and SLOW fusion installed-capacity scenarios in Table 5.
#
# Main assumptions:
#   - Table 5 installed capacity is interpreted as electric capacity [GWe].
#   - ARC net electric power is 190 MWe per plant.
#   - The required number of ARC plants is rounded up using ceil().
#   - Startup material demand occurs in the year when new ARC plants are added.
#   - Annual Li-6 burnup is converted to Li-90 equivalent by dividing by 0.90.
#   - Be replacement demand occurs every fixed interval after plant startup.
#   - REBCO demand is treated as startup-only.
#
# Notes:
#   - The initial 2050 installed capacity is treated as plants newly built in 2050.
#   - Operation demand is counted for the full year for plants present in that year.
#   - Small year-to-year wiggles in REBCO annual demand are caused by integer
#     plant rounding using ceil() on a linearly interpolated installed-capacity target.
#
# Outputs:
#   - Console summary
#   - Installed-capacity and ARC plant-count plot
#   - Annual new ARC builds plot
#   - Annual Be demand plots for 5-year and 10-year replacement intervals
#   - Annual Li-90 and REBCO demand plots
#   - Optional CSV summary
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

# Be replacement cases to compare.
BE_REPLACEMENT_INTERVALS_YR = [5, 10]

# Be plotting mode:
#   "separate_same_ylim" : one Be figure for 5-year replacement and one for 10-year replacement,
#                          with the same y-axis range for direct visual comparison.
#   "combined"           : one Be figure containing all four cases:
#                          FAST-5yr, SLOW-5yr, FAST-10yr, SLOW-10yr.
BE_PLOT_MODE = "separate_same_ylim"

SAVE_CSV = False
OUT_CSV = "demand_response_Fusion_Reactors_v1_summary.csv"

# If True, annual demand is also shown as thin vertical yearly spikes.
# For a cleaner paper-style figure, False is often preferable.
USE_VERTICAL_SPIKES = True

# Axis padding fraction for automatic y-axis limits.
Y_AXIS_PAD_FRACTION = 0.08


# ============================================================
# ARC PARAMETERS
# ============================================================

ARC_NET_ELECTRIC_MWE = 190.0
ARC_NET_ELECTRIC_GWE = ARC_NET_ELECTRIC_MWE / 1000.0

ARC_BE_STARTUP_T_PER_PLANT = 88.7
ARC_LI90_STARTUP_T_PER_PLANT = 132.6
ARC_REBCO_KM_PER_PLANT = 5730.0

ARC_BE_REPLACEMENT_T_PER_PLANT = 3.82

ARC_LI6_BURN_KG_PER_YR_PER_PLANT = 57.8
ARC_LI90_ENRICHMENT = 0.90

ARC_LI90_OPERATION_T_PER_YR_PER_PLANT = (
    ARC_LI6_BURN_KG_PER_YR_PER_PLANT
    / ARC_LI90_ENRICHMENT
    / 1000.0
)


# ============================================================
# Table 5 deployment targets
# ============================================================

# Installed capacity is interpreted as electric capacity [GWe].
TABLE5_FAST_GWE = {
    2050: 3.0,
    2060: 11.0,
    2070: 38.0,
    2080: 152.0,
    2090: 395.0,
    2100: 1024.0,
}

TABLE5_SLOW_GWE = {
    2050: 1.0,
    2060: 4.0,
    2070: 14.0,
    2080: 27.0,
    2090: 71.0,
    2100: 184.0,
}


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "legend.frameon": False,
})

COLOR_FAST = "tab:blue"
COLOR_SLOW = "tab:green"


# ============================================================
# Utility functions
# ============================================================

def human_format(x, pos):
    """
    Format large axis numbers in a compact way.
    """
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def nice_upper_limit(values, pad_fraction=0.08):
    """
    Return a visually clean upper y-axis limit.
    """
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


def set_common_axis_style(ax, ylabel, title, ylim_max=None):
    """
    Apply common formatting to annual-demand plots.
    """
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))

    if ylim_max is not None:
        ax.set_ylim(0.0, ylim_max)


def interpolate_capacity_targets(target_dict, start_year, end_year):
    """
    Linearly interpolate Table 5 installed-capacity targets
    to obtain an annual installed-capacity trajectory.

    The original Table 5 values are given every 10 years.
    This function creates annual values between the tabulated points.
    """
    years_known = np.array(sorted(target_dict.keys()), dtype=float)
    values_known = np.array([target_dict[int(y)] for y in years_known], dtype=float)

    years = np.arange(start_year, end_year + 1, dtype=int)
    values = np.interp(years, years_known, values_known)

    return pd.DataFrame({
        "year": years,
        "target_capacity_gwe": values,
    })


def required_plants_from_capacity(capacity_gwe):
    """
    Convert installed electric capacity target [GWe]
    to the required number of ARC plants.
    """
    return int(math.ceil(capacity_gwe / ARC_NET_ELECTRIC_GWE))


# ============================================================
# Calculation functions
# ============================================================

def add_be_replacement_demand(df, replacement_interval_yr):
    """
    Add Be replacement demand for a selected replacement interval.

    Each yearly cohort of new builds receives replacement material every
    replacement_interval_yr years after startup.
    """
    out = df.copy()
    replacement_demand = []

    for year in out["year"]:
        demand_this_year = 0.0

        for _, cohort in out.iterrows():
            build_year = int(cohort["year"])
            n_built = int(cohort["arc_new_builds"])

            if n_built <= 0:
                continue

            years_since_build = int(year - build_year)

            if years_since_build <= 0:
                continue

            if years_since_build % replacement_interval_yr == 0:
                demand_this_year += n_built * ARC_BE_REPLACEMENT_T_PER_PLANT

        replacement_demand.append(demand_this_year)

    out["be_replacement_interval_yr"] = replacement_interval_yr
    out["be_replacement_demand_t"] = replacement_demand
    out["be_total_annual_demand_t"] = (
        out["be_startup_demand_t"] + out["be_replacement_demand_t"]
    )
    out["be_cumulative_demand_t"] = out["be_total_annual_demand_t"].cumsum()

    return out


def build_arc_base_scenario(target_dict, scenario_name):
    """
    Build annual ARC deployment and material demand components that do not
    depend on the Be replacement interval.

    New plants are added whenever the rounded-up plant requirement
    increases relative to the previous year.
    """
    df = interpolate_capacity_targets(
        target_dict=target_dict,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    required_plants = []
    new_builds = []

    previous_required = 0

    for _, row in df.iterrows():
        n_required = required_plants_from_capacity(row["target_capacity_gwe"])
        n_new = max(n_required - previous_required, 0)

        required_plants.append(n_required)
        new_builds.append(n_new)

        previous_required = n_required

    df["scenario"] = scenario_name
    df["arc_required_plants"] = required_plants
    df["arc_new_builds"] = new_builds

    # Startup material demands.
    df["be_startup_demand_t"] = (
        df["arc_new_builds"] * ARC_BE_STARTUP_T_PER_PLANT
    )

    df["li90_startup_demand_t"] = (
        df["arc_new_builds"] * ARC_LI90_STARTUP_T_PER_PLANT
    )

    df["rebco_startup_demand_km"] = (
        df["arc_new_builds"] * ARC_REBCO_KM_PER_PLANT
    )

    # Annual operation Li demand.
    # This is based on the total number of plants operating in that year.
    df["li90_operation_demand_t"] = (
        df["arc_required_plants"] * ARC_LI90_OPERATION_T_PER_YR_PER_PLANT
    )

    # Total annual Li-90 demand.
    df["li90_total_annual_demand_t"] = (
        df["li90_startup_demand_t"] + df["li90_operation_demand_t"]
    )

    # REBCO is treated as startup-only.
    df["rebco_total_annual_demand_km"] = df["rebco_startup_demand_km"]

    # Cumulative material demands that do not depend on Be replacement.
    df["li90_cumulative_demand_t"] = df["li90_total_annual_demand_t"].cumsum()
    df["rebco_cumulative_demand_km"] = df["rebco_total_annual_demand_km"].cumsum()

    return df


def print_scenario_summary(df):
    """
    Print a compact summary for the final year.
    """
    last = df.iloc[-1]
    scenario = last["scenario"]

    print("------------------------------------------------------------")
    print(f"[{scenario}] final year summary")
    print(f"Year                         : {int(last['year'])}")
    print(f"Target installed capacity    : {last['target_capacity_gwe']:.1f} GWe")
    print(f"Required ARC plants          : {int(last['arc_required_plants'])}")
    print(f"Final-year new ARC builds    : {int(last['arc_new_builds'])}")
    print(f"Cumulative Li-90 demand      : {last['li90_cumulative_demand_t']:,.1f} t")
    print(f"Cumulative REBCO demand      : {last['rebco_cumulative_demand_km']:,.0f} km")
    print("------------------------------------------------------------")


def print_be_summary(df):
    """
    Print Be cumulative demand for a selected Be replacement interval.
    """
    last = df.iloc[-1]
    scenario = last["scenario"]
    interval = int(last["be_replacement_interval_yr"])

    print("------------------------------------------------------------")
    print(f"[{scenario}] Be summary, {interval}-year replacement")
    print(f"Cumulative Be demand         : {last['be_cumulative_demand_t']:,.1f} t")
    print(f"Final-year annual Be demand  : {last['be_total_annual_demand_t']:,.1f} t")
    print("------------------------------------------------------------")


# ============================================================
# Plotting functions
# ============================================================

def draw_fast_slow_lines(ax, fast_df, slow_df, y_col):
    """
    Draw FAST and SLOW annual demand curves on a given axis.
    """
    if USE_VERTICAL_SPIKES:
        ax.vlines(
            fast_df["year"],
            0.0,
            fast_df[y_col],
            color=COLOR_FAST,
            alpha=0.35,
            linewidth=1.0,
        )
        ax.vlines(
            slow_df["year"],
            0.0,
            slow_df[y_col],
            color=COLOR_SLOW,
            alpha=0.35,
            linewidth=1.0,
        )

    ax.plot(
        fast_df["year"],
        fast_df[y_col],
        color=COLOR_FAST,
        label="FAST",
    )

    ax.plot(
        slow_df["year"],
        slow_df[y_col],
        color=COLOR_SLOW,
        linestyle="--",
        label="SLOW",
    )


def plot_annual_demand(
    fast_df,
    slow_df,
    y_col,
    ylabel,
    title,
    ylim_max=None,
):
    """
    Plot annual demand for FAST and SLOW scenarios.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    draw_fast_slow_lines(ax, fast_df, slow_df, y_col)

    set_common_axis_style(
        ax=ax,
        ylabel=ylabel,
        title=title,
        ylim_max=ylim_max,
    )

    fig.tight_layout()
    plt.show()


def plot_combined_be_demand(be_results):
    """
    Plot all four Be cases in a single figure:
    FAST-5yr, SLOW-5yr, FAST-10yr, and SLOW-10yr.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for interval, fast_df, slow_df in be_results:
        ax.plot(
            fast_df["year"],
            fast_df["be_total_annual_demand_t"],
            color=COLOR_FAST,
            linestyle="-" if interval == 5 else ":",
            label=f"FAST, {interval}-yr replacement",
        )

        ax.plot(
            slow_df["year"],
            slow_df["be_total_annual_demand_t"],
            color=COLOR_SLOW,
            linestyle="--" if interval == 5 else "-.",
            label=f"SLOW, {interval}-yr replacement",
        )

    all_values = []
    for _, fast_df, slow_df in be_results:
        all_values.extend(fast_df["be_total_annual_demand_t"].values)
        all_values.extend(slow_df["be_total_annual_demand_t"].values)

    ylim_max = nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION)

    set_common_axis_style(
        ax=ax,
        ylabel="Annual Be demand (t)",
        title="ARC Be Annual Demand: 5-year vs 10-year Replacement",
        ylim_max=ylim_max,
    )

    fig.tight_layout()
    plt.show()


def plot_installed_capacity_and_plants(fast_df, slow_df):
    """
    Plot the target installed capacity and the corresponding ARC plant count.

    This function intentionally keeps the original dual-axis plotting style.
    The left axis shows Table 5 installed electric capacity [GWe], and the
    right axis shows the corresponding required number of ARC plants.
    """
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))

    ax1.plot(
        fast_df["year"],
        fast_df["target_capacity_gwe"],
        color=COLOR_FAST,
        label="FAST capacity",
    )
    ax1.plot(
        slow_df["year"],
        slow_df["target_capacity_gwe"],
        color=COLOR_SLOW,
        linestyle="--",
        label="SLOW capacity",
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Installed electric capacity target (GWe)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(human_format))

    ax2 = ax1.twinx()
    ax2.step(
        fast_df["year"],
        fast_df["arc_required_plants"],
        where="post",
        color=COLOR_FAST,
        alpha=0.35,
        label="FAST ARC plants",
    )
    ax2.step(
        slow_df["year"],
        slow_df["arc_required_plants"],
        where="post",
        color=COLOR_SLOW,
        linestyle="--",
        alpha=0.35,
        label="SLOW ARC plants",
    )

    ax2.set_ylabel("Required ARC plants")
    ax2.yaxis.set_major_formatter(FuncFormatter(human_format))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax1.set_title("Table 5 Fusion Deployment Targets Converted to ARC Plants")

    fig.tight_layout()
    plt.show()


def plot_arc_new_builds(fast_df, slow_df):
    """
    Plot annual new ARC builds.

    This plot is useful for diagnosing small wiggles in REBCO annual demand,
    because REBCO annual demand is directly proportional to annual new builds.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    if USE_VERTICAL_SPIKES:
        ax.vlines(
            fast_df["year"],
            0.0,
            fast_df["arc_new_builds"],
            color=COLOR_FAST,
            alpha=0.35,
            linewidth=1.0,
        )
        ax.vlines(
            slow_df["year"],
            0.0,
            slow_df["arc_new_builds"],
            color=COLOR_SLOW,
            alpha=0.35,
            linewidth=1.0,
        )

    ax.step(
        fast_df["year"],
        fast_df["arc_new_builds"],
        where="mid",
        color=COLOR_FAST,
        label="FAST",
    )

    ax.step(
        slow_df["year"],
        slow_df["arc_new_builds"],
        where="mid",
        color=COLOR_SLOW,
        linestyle="--",
        label="SLOW",
    )

    new_build_ylim = nice_upper_limit(
        list(fast_df["arc_new_builds"].values)
        + list(slow_df["arc_new_builds"].values),
        Y_AXIS_PAD_FRACTION,
    )

    set_common_axis_style(
        ax=ax,
        ylabel="New ARC builds per year",
        title="Annual New ARC Builds",
        ylim_max=new_build_ylim,
    )

    fig.tight_layout()
    plt.show()


# ============================================================
# Main driver
# ============================================================

def main():
    print("")
    print("============================================================")
    print("ARC material demand calculation from Table 5 scenarios")
    print("============================================================")
    print(f"ARC net electric power      : {ARC_NET_ELECTRIC_MWE:.1f} MWe")
    print(f"Li-90 operation demand/ARC  : {ARC_LI90_OPERATION_T_PER_YR_PER_PLANT:.5f} t/yr")
    print(f"Be replacement intervals    : {BE_REPLACEMENT_INTERVALS_YR}")
    print(f"Be plot mode                : {BE_PLOT_MODE}")
    print("============================================================")
    print("")

    fast_base = build_arc_base_scenario(TABLE5_FAST_GWE, "FAST")
    slow_base = build_arc_base_scenario(TABLE5_SLOW_GWE, "SLOW")

    print_scenario_summary(fast_base)
    print_scenario_summary(slow_base)

    all_outputs = []
    be_results = []

    for interval in BE_REPLACEMENT_INTERVALS_YR:
        fast_be = add_be_replacement_demand(fast_base, interval)
        slow_be = add_be_replacement_demand(slow_base, interval)

        print_be_summary(fast_be)
        print_be_summary(slow_be)

        all_outputs.append(fast_be)
        all_outputs.append(slow_be)
        be_results.append((interval, fast_be, slow_be))

    combined = pd.concat(all_outputs, ignore_index=True)

    if SAVE_CSV:
        combined.to_csv(OUT_CSV, index=False)
        print(f"[INFO] Wrote: {OUT_CSV}")

    # Common y-axis limit for Be figures across all replacement intervals.
    be_all_values = []
    for _, fast_be, slow_be in be_results:
        be_all_values.extend(fast_be["be_total_annual_demand_t"].values)
        be_all_values.extend(slow_be["be_total_annual_demand_t"].values)

    be_common_ylim = nice_upper_limit(be_all_values, Y_AXIS_PAD_FRACTION)

    li90_ylim = nice_upper_limit(
        list(fast_base["li90_total_annual_demand_t"].values)
        + list(slow_base["li90_total_annual_demand_t"].values),
        Y_AXIS_PAD_FRACTION,
    )

    rebco_ylim = nice_upper_limit(
        list(fast_base["rebco_total_annual_demand_km"].values)
        + list(slow_base["rebco_total_annual_demand_km"].values),
        Y_AXIS_PAD_FRACTION,
    )

    plot_installed_capacity_and_plants(fast_base, slow_base)
    plot_arc_new_builds(fast_base, slow_base)

    if BE_PLOT_MODE == "combined":
        plot_combined_be_demand(be_results)
    elif BE_PLOT_MODE == "separate_same_ylim":
        for interval, fast_be, slow_be in be_results:
            plot_annual_demand(
                fast_df=fast_be,
                slow_df=slow_be,
                y_col="be_total_annual_demand_t",
                ylabel="Annual Be demand (t)",
                title=(
                    "ARC Be Annual Demand "
                    f"(startup + {interval}-year replacement)"
                ),
                ylim_max=be_common_ylim,
            )
    else:
        raise ValueError(
            "Invalid BE_PLOT_MODE. Use 'separate_same_ylim' or 'combined'."
        )

    plot_annual_demand(
        fast_df=fast_base,
        slow_df=slow_base,
        y_col="li90_total_annual_demand_t",
        ylabel="Annual Li-90 demand (t)",
        title="ARC Li-90 Annual Demand (startup + operation)",
        ylim_max=li90_ylim,
    )

    plot_annual_demand(
        fast_df=fast_base,
        slow_df=slow_base,
        y_col="rebco_total_annual_demand_km",
        ylabel="Annual REBCO demand (km)",
        title="ARC REBCO Annual Demand (startup only)",
        ylim_max=rebco_ylim,
    )


if __name__ == "__main__":
    main()
