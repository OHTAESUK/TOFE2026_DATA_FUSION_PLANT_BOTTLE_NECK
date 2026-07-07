#!/usr/bin/env python3
# ============================================================
# demand_response_Fusion_Reactors_compare_v2_combined_plots.py
#
# Material annual demand comparison for ARC, Hammir, and EU-DEMO WCLL
# under Table 5 fusion deployment scenarios.
#
# Modified from v1:
#   1) Actual enriched lithium demand and Hammir Li90-equivalent demand
#      are plotted in the same figure.
#      - Actual enriched Li: solid lines
#      - Hammir Li90-equivalent: dashed line with the same Hammir color
#
#   2) Hammir WB and EU-DEMO WCLL Nb3Sn annual demand are plotted
#      in one figure using two y-axes.
#
# Main assumptions:
#   - Table 5 installed capacity is interpreted as electric capacity [GWe].
#   - Required plant count is rounded up using ceil().
#   - START_YEAR installed capacity is treated as the baseline fleet.
#   - Startup material demand is counted only for incremental new builds after START_YEAR.
#   - Annual Li-6 burnup is converted to enriched lithium mass using
#     the reactor-specific Li-6 enrichment.
#   - Hammir uses 30%-enriched lithium. For comparison with ARC/EU-DEMO,
#     Hammir Li-30 demand is converted to Li-90-equivalent demand using:
#
#         Li90-to-Li30 factor = (0.90 - 0.075) / (0.30 - 0.075) = 3.6667
#         Hammir Li90-equivalent demand = Hammir Li30 demand / 3.6667
#
#   - ARC and EU-DEMO already use Li-90, so actual enriched-Li demand and
#     Li90-equivalent demand are identical for those reactors.
#   - ARC Be replacement demand occurs every selected interval after startup.
#   - Hammir WB, EU-DEMO Nb3Sn, and all REBCO demands are startup-only.
#
# Outputs:
#   - Console summary
#   - Table 5 installed-capacity target plot
#   - Required plant-count and annual new-build comparison plots
#   - Annual REBCO plot
#   - Combined annual enriched-Li plot:
#       actual enriched Li for all reactors + Hammir Li90-equivalent
#   - Combined annual WB/Nb3Sn plot with dual y-axes
#   - ARC annual Be demand plots for 5-year and 10-year replacement intervals
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

SCENARIOS_TO_PLOT = ["FAST", "SLOW"]
BE_REPLACEMENT_INTERVALS_YR = [5, 10]

# Be plotting mode:
#   "separate_same_ylim" : one figure for 5-year replacement and one for 10-year replacement,
#                          with the same y-axis range.
#   "combined"           : one figure containing FAST/SLOW and 5-year/10-year cases.
BE_PLOT_MODE = "separate_same_ylim"

SAVE_CSV = False
OUT_CSV = "demand_response_Fusion_Reactors_compare_v2_combined_plots_summary.csv"

# If True, annual demand is also shown as thin vertical yearly spikes.
USE_VERTICAL_SPIKES = True

Y_AXIS_PAD_FRACTION = 0.08


# ============================================================
# Table 5 deployment targets
# ============================================================

# Installed capacity is interpreted as electric capacity [GWe].
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
# Reactor parameters
# ============================================================

REACTORS = {
    "Compact Tokamak": {
        "net_electric_mwe": 190.0,
        "rebco_km_per_plant": 5730.0,
        "li_startup_t_per_plant": 132.6,
        "li_enrichment": 0.90,
        "li6_burn_kg_per_yr": 57.8,
        "be_startup_t_per_plant": 88.7,
        "be_replacement_t_per_plant": 3.82,
        "wb_startup_t_per_plant": 0.0,
        "nb3sn_startup_t_per_plant": 0.0,
    },
    "Tandem Mirror": {
        "net_electric_mwe": 165.0,
        "rebco_km_per_plant": 1406.0,
        "li_startup_t_per_plant": 20.1*2,
        "li_enrichment": 0.30,
        "li6_burn_kg_per_yr": 38.5,
        "be_startup_t_per_plant": 0.0,
        "be_replacement_t_per_plant": 0.0,
        "wb_startup_t_per_plant": 781.18,
        "nb3sn_startup_t_per_plant": 0.0,
    },
    "Large-scale Tokamak": {
        "net_electric_mwe": 1550.0,
        "rebco_km_per_plant": 714.0,
        "li_startup_t_per_plant": 133.4,
        "li_enrichment": 0.90,
        "li6_burn_kg_per_yr": 550.0,
        "be_startup_t_per_plant": 0.0,
        "be_replacement_t_per_plant": 0.0,
        "wb_startup_t_per_plant": 0.0,
        "nb3sn_startup_t_per_plant": 595.0,
    },
}


# ============================================================
# Lithium blending constants
# ============================================================

LI90_ENRICHMENT = 0.90
LI30_TARGET_ENRICHMENT = 0.30
LI_NAT_ENRICHMENT = 0.075

LI90_TO_LI30_FACTOR = (
    (LI90_ENRICHMENT - LI_NAT_ENRICHMENT)
    / (LI30_TARGET_ENRICHMENT - LI_NAT_ENRICHMENT)
)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "legend.frameon": False,
})

REACTOR_COLORS = {
    "Compact Tokamak": "tab:blue",
    "Tandem Mirror": "tab:orange",
    "Large-scale Tokamak": "tab:green",
}

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
    """Format large axis numbers in a compact way."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def nice_upper_limit(values, pad_fraction=0.08):
    """Return a visually clean upper y-axis limit."""
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
    """Apply common formatting to annual-demand plots."""
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
    """
    years_known = np.array(sorted(target_dict.keys()), dtype=float)
    values_known = np.array([target_dict[int(y)] for y in years_known], dtype=float)

    years = np.arange(start_year, end_year + 1, dtype=int)
    values = np.interp(years, years_known, values_known)

    return pd.DataFrame({
        "year": years,
        "target_capacity_gwe": values,
    })


def required_plants_from_capacity(capacity_gwe, reactor_name):
    """Convert installed electric capacity [GWe] to required plant count."""
    net_electric_gwe = REACTORS[reactor_name]["net_electric_mwe"] / 1000.0
    return int(math.ceil(capacity_gwe / net_electric_gwe))


def li90_equivalent_demand_t(reactor_name, actual_enriched_li_demand_t):
    """
    Convert actual enriched lithium demand to Li90-equivalent demand.

    ARC and EU-DEMO use Li90 directly. Hammir uses Li30, so Li30 demand
    is divided by the Li90-to-Li30 blending factor.
    """
    if reactor_name == "Tandem Mirror":
        return actual_enriched_li_demand_t / LI90_TO_LI30_FACTOR
    return actual_enriched_li_demand_t


# ============================================================
# Calculation functions
# ============================================================

def build_reactor_base_scenario(target_dict, scenario_name, reactor_name):
    """
    Build annual deployment and material demand components for one reactor
    and one deployment scenario.
    """
    params = REACTORS[reactor_name]
    df = interpolate_capacity_targets(target_dict, START_YEAR, END_YEAR)

    required_plants = []
    new_builds = []

    # Treat START_YEAR installed capacity as the baseline fleet.
    # This avoids an artificial startup/new-build spike at the first plotted year.
    # Material demand is therefore interpreted as incremental demand after START_YEAR.
    previous_required = None

    for _, row in df.iterrows():
        n_required = required_plants_from_capacity(row["target_capacity_gwe"], reactor_name)

        if previous_required is None:
            n_new = 0
        else:
            n_new = max(n_required - previous_required, 0)

        required_plants.append(n_required)
        new_builds.append(n_new)
        previous_required = n_required

    df["scenario"] = scenario_name
    df["reactor"] = reactor_name
    df["required_plants"] = required_plants
    df["new_builds"] = new_builds

    # Startup-only material demands.
    df["rebco_annual_demand_km"] = df["new_builds"] * params["rebco_km_per_plant"]
    df["li_startup_annual_demand_t"] = df["new_builds"] * params["li_startup_t_per_plant"]
    df["wb_annual_demand_t"] = df["new_builds"] * params["wb_startup_t_per_plant"]
    df["nb3sn_annual_demand_t"] = df["new_builds"] * params["nb3sn_startup_t_per_plant"]

    # Annual operation lithium demand using reactor-specific enrichment.
    li_operation_t_per_yr_per_plant = (
        params["li6_burn_kg_per_yr"] / params["li_enrichment"] / 1000.0
    )
    df["li_operation_annual_demand_t"] = (
        df["required_plants"] * li_operation_t_per_yr_per_plant
    )

    # Actual enriched lithium demand.
    # ARC and EU-DEMO: Li90. Hammir: Li30.
    df["li_actual_annual_demand_t"] = (
        df["li_startup_annual_demand_t"] + df["li_operation_annual_demand_t"]
    )

    # Li90-equivalent demand for cross-reactor comparison.
    df["li90_equiv_annual_demand_t"] = df["li_actual_annual_demand_t"].apply(
        lambda x: li90_equivalent_demand_t(reactor_name, x)
    )

    # Cumulative demands.
    df["rebco_cumulative_demand_km"] = df["rebco_annual_demand_km"].cumsum()
    df["li_actual_cumulative_demand_t"] = df["li_actual_annual_demand_t"].cumsum()
    df["li90_equiv_cumulative_demand_t"] = df["li90_equiv_annual_demand_t"].cumsum()
    df["wb_cumulative_demand_t"] = df["wb_annual_demand_t"].cumsum()
    df["nb3sn_cumulative_demand_t"] = df["nb3sn_annual_demand_t"].cumsum()

    return df


def add_arc_be_replacement_demand(df, replacement_interval_yr):
    """Add ARC Be startup and replacement demand for a selected interval."""
    if df["reactor"].iloc[0] != "Compact Tokamak":
        raise ValueError("Be replacement demand is only defined for Compact Tokamak in this script.")

    params = REACTORS["Compact Tokamak"]
    out = df.copy()

    out["be_startup_annual_demand_t"] = out["new_builds"] * params["be_startup_t_per_plant"]

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
                demand_this_year += n_built * params["be_replacement_t_per_plant"]
        replacement_demand.append(demand_this_year)

    out["be_replacement_interval_yr"] = replacement_interval_yr
    out["be_replacement_annual_demand_t"] = replacement_demand
    out["be_total_annual_demand_t"] = (
        out["be_startup_annual_demand_t"] + out["be_replacement_annual_demand_t"]
    )
    out["be_cumulative_demand_t"] = out["be_total_annual_demand_t"].cumsum()

    return out


def build_all_base_scenarios():
    """Build base annual-demand data for all reactors and scenarios."""
    frames = []
    for scenario_name in SCENARIOS_TO_PLOT:
        target_dict = TABLE5_TARGETS_GWE[scenario_name]
        for reactor_name in REACTORS:
            frames.append(build_reactor_base_scenario(target_dict, scenario_name, reactor_name))
    return pd.concat(frames, ignore_index=True)


def print_summary(base_df, arc_be_results):
    """Print compact final-year summaries."""
    print("")
    print("============================================================")
    print("Fusion reactor material demand comparison")
    print("============================================================")
    print(f"Li90-to-Li30 blending factor : {LI90_TO_LI30_FACTOR:.4f}")
    print("Tandem Mirror Li90-equivalent demand = Li30 demand / factor")
    print("============================================================")
    print("")

    for scenario_name in SCENARIOS_TO_PLOT:
        for reactor_name in REACTORS:
            sub = base_df[(base_df["scenario"] == scenario_name) & (base_df["reactor"] == reactor_name)]
            last = sub.iloc[-1]
            print("------------------------------------------------------------")
            print(f"[{scenario_name}] {reactor_name}")
            print(f"Final year target capacity     : {last['target_capacity_gwe']:.1f} GWe")
            print(f"Required plants                : {int(last['required_plants'])}")
            print(f"Final-year new builds          : {int(last['new_builds'])}")
            print(f"Cumulative REBCO               : {last['rebco_cumulative_demand_km']:,.0f} km")
            print(f"Cumulative actual enriched Li  : {last['li_actual_cumulative_demand_t']:,.1f} t")
            print(f"Cumulative Li90-equivalent     : {last['li90_equiv_cumulative_demand_t']:,.1f} t")
            if reactor_name == "Tandem Mirror":
                print(f"Cumulative WB                  : {last['wb_cumulative_demand_t']:,.1f} t")
            if reactor_name == "Large-scale Tokamak":
                print(f"Cumulative Nb3Sn               : {last['nb3sn_cumulative_demand_t']:,.1f} t")
            print("------------------------------------------------------------")

    for interval, df in arc_be_results:
        for scenario_name in SCENARIOS_TO_PLOT:
            sub = df[df["scenario"] == scenario_name]
            last = sub.iloc[-1]
            print("------------------------------------------------------------")
            print(f"[{scenario_name}] Compact Tokamak Be, {interval}-year replacement")
            print(f"Cumulative Be demand           : {last['be_cumulative_demand_t']:,.1f} t")
            print(f"Final-year annual Be demand    : {last['be_total_annual_demand_t']:,.1f} t")
            print("------------------------------------------------------------")


# ============================================================
# Plotting functions
# ============================================================

def plot_capacity_targets():
    """Plot Table 5 installed-capacity targets."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    all_values = []
    for scenario_name in SCENARIOS_TO_PLOT:
        df = interpolate_capacity_targets(TABLE5_TARGETS_GWE[scenario_name], START_YEAR, END_YEAR)
        all_values.extend(df["target_capacity_gwe"].values)
        ax.plot(
            df["year"],
            df["target_capacity_gwe"],
            color=SCENARIO_COLORS[scenario_name],
            linestyle=SCENARIO_LINESTYLES[scenario_name],
            label=f"{scenario_name} target",
        )

    set_common_axis_style(
        ax=ax,
        ylabel="Installed electric capacity target (GWe)",
        title="Table 5 Fusion Installed-Capacity Targets",
        ylim_max=nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION),
    )
    fig.tight_layout()
    plt.show()


def plot_reactor_comparison(base_df, scenario_name, y_col, ylabel, title):
    """Plot annual demand comparison across reactors for one deployment scenario."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_values = []

    for reactor_name in REACTORS:
        sub = base_df[(base_df["scenario"] == scenario_name) & (base_df["reactor"] == reactor_name)].copy()
        all_values.extend(sub[y_col].values)

        if USE_VERTICAL_SPIKES:
            ax.vlines(
                sub["year"],
                0.0,
                sub[y_col],
                color=REACTOR_COLORS[reactor_name],
                alpha=0.22,
                linewidth=1.0,
            )

        ax.plot(
            sub["year"],
            sub[y_col],
            color=REACTOR_COLORS[reactor_name],
            label=reactor_name,
        )

    set_common_axis_style(
        ax=ax,
        ylabel=ylabel,
        title=f"{title} ({scenario_name})",
        ylim_max=nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION),
    )
    fig.tight_layout()
    plt.show()


def plot_lithium_actual_with_hammir_equiv(base_df, scenario_name):
    """
    Plot annual actual enriched lithium demand for all reactors.
    Hammir Li90-equivalent demand is overlaid as a dashed line using
    the same Hammir color.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_values = []

    for reactor_name in REACTORS:
        sub = base_df[
            (base_df["scenario"] == scenario_name)
            & (base_df["reactor"] == reactor_name)
        ].copy()

        all_values.extend(sub["li_actual_annual_demand_t"].values)

        if USE_VERTICAL_SPIKES:
            ax.vlines(
                sub["year"],
                0.0,
                sub["li_actual_annual_demand_t"],
                color=REACTOR_COLORS[reactor_name],
                alpha=0.22,
                linewidth=1.0,
            )

        ax.plot(
            sub["year"],
            sub["li_actual_annual_demand_t"],
            color=REACTOR_COLORS[reactor_name],
            linestyle="-",
            label=f"{reactor_name} actual",
        )

        if reactor_name == "Tandem Mirror":
            all_values.extend(sub["li90_equiv_annual_demand_t"].values)

            ax.plot(
                sub["year"],
                sub["li90_equiv_annual_demand_t"],
                color=REACTOR_COLORS[reactor_name],
                linestyle="--",
                label="Tandem Mirror Li90-equivalent",
            )

    set_common_axis_style(
        ax=ax,
        ylabel="Annual enriched lithium demand (t)",
        title=f"Annual Enriched Lithium Demand ({scenario_name})",
        ylim_max=nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION),
    )
    fig.tight_layout()
    plt.show()


def plot_wb_nb3sn_dual_axis(base_df, scenario_name):
    """
    Plot Hammir WB and EU-DEMO WCLL Nb3Sn annual demands in one figure
    using separate y-axes.
    """
    h_sub = base_df[
        (base_df["scenario"] == scenario_name)
        & (base_df["reactor"] == "Tandem Mirror")
    ].copy()

    e_sub = base_df[
        (base_df["scenario"] == scenario_name)
        & (base_df["reactor"] == "Large-scale Tokamak")
    ].copy()

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))

    # Left axis: Hammir WB
    if USE_VERTICAL_SPIKES:
        ax1.vlines(
            h_sub["year"],
            0.0,
            h_sub["wb_annual_demand_t"],
            color=REACTOR_COLORS["Tandem Mirror"],
            alpha=0.25,
            linewidth=1.0,
        )

    ax1.plot(
        h_sub["year"],
        h_sub["wb_annual_demand_t"],
        color=REACTOR_COLORS["Tandem Mirror"],
        linestyle="-",
        label="Tandem Mirror WB",
    )

    ax1.set_xlim(START_YEAR, END_YEAR)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual WB demand (t)", color=REACTOR_COLORS["Tandem Mirror"])
    ax1.tick_params(axis="y", labelcolor=REACTOR_COLORS["Tandem Mirror"])
    ax1.yaxis.set_major_formatter(FuncFormatter(human_format))
    ax1.set_ylim(
        0.0,
        nice_upper_limit(h_sub["wb_annual_demand_t"].values, Y_AXIS_PAD_FRACTION),
    )
    ax1.grid(True, alpha=0.3)

    # Right axis: EU-DEMO Nb3Sn
    ax2 = ax1.twinx()

    if USE_VERTICAL_SPIKES:
        ax2.vlines(
            e_sub["year"],
            0.0,
            e_sub["nb3sn_annual_demand_t"],
            color=REACTOR_COLORS["Large-scale Tokamak"],
            alpha=0.25,
            linewidth=1.0,
        )

    ax2.plot(
        e_sub["year"],
        e_sub["nb3sn_annual_demand_t"],
        color=REACTOR_COLORS["Large-scale Tokamak"],
        linestyle="--",
        label=r"Large-scale Tokamak Nb$_3$Sn",
    )

    ax2.set_ylabel(r"Annual Nb$_3$Sn demand (t)", color=REACTOR_COLORS["Large-scale Tokamak"])
    ax2.tick_params(axis="y", labelcolor=REACTOR_COLORS["Large-scale Tokamak"])
    ax2.yaxis.set_major_formatter(FuncFormatter(human_format))
    ax2.set_ylim(
        0.0,
        nice_upper_limit(e_sub["nb3sn_annual_demand_t"].values, Y_AXIS_PAD_FRACTION),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )

    ax1.set_title(rf"Annual WB and Nb$_3$Sn Demand ({scenario_name})")
    fig.tight_layout()
    plt.show()


def plot_arc_be_demand(arc_be_results):
    """Plot ARC Be annual demand for selected replacement intervals."""
    if BE_PLOT_MODE == "combined":
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        all_values = []

        for interval, df in arc_be_results:
            for scenario_name in SCENARIOS_TO_PLOT:
                sub = df[df["scenario"] == scenario_name].copy()
                all_values.extend(sub["be_total_annual_demand_t"].values)
                ax.plot(
                    sub["year"],
                    sub["be_total_annual_demand_t"],
                    color=SCENARIO_COLORS[scenario_name],
                    linestyle="-" if interval == 5 else ":",
                    label=f"{scenario_name}, {interval}-yr replacement",
                )

        set_common_axis_style(
            ax=ax,
            ylabel="Annual Be demand (t)",
            title="Compact Tokamak Be Annual Demand: 5-year vs 10-year Replacement",
            ylim_max=nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION),
        )
        fig.tight_layout()
        plt.show()

    elif BE_PLOT_MODE == "separate_same_ylim":
        all_values = []
        for _, df in arc_be_results:
            all_values.extend(df["be_total_annual_demand_t"].values)
        ylim_max = nice_upper_limit(all_values, Y_AXIS_PAD_FRACTION)

        for interval, df in arc_be_results:
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            for scenario_name in SCENARIOS_TO_PLOT:
                sub = df[df["scenario"] == scenario_name].copy()
                if USE_VERTICAL_SPIKES:
                    ax.vlines(
                        sub["year"],
                        0.0,
                        sub["be_total_annual_demand_t"],
                        color=SCENARIO_COLORS[scenario_name],
                        alpha=0.25,
                        linewidth=1.0,
                    )
                ax.plot(
                    sub["year"],
                    sub["be_total_annual_demand_t"],
                    color=SCENARIO_COLORS[scenario_name],
                    linestyle=SCENARIO_LINESTYLES[scenario_name],
                    label=scenario_name,
                )

            set_common_axis_style(
                ax=ax,
                ylabel="Annual Be demand (t)",
                title=f"Compact Tokamak Be Annual Demand (startup + {interval}-year replacement)",
                ylim_max=ylim_max,
            )
            fig.tight_layout()
            plt.show()
    else:
        raise ValueError("Invalid BE_PLOT_MODE. Use 'separate_same_ylim' or 'combined'.")


# ============================================================
# Main driver
# ============================================================

def main():
    base_df = build_all_base_scenarios()

    arc_base = base_df[base_df["reactor"] == "Compact Tokamak"].copy()
    arc_be_results = []

    for interval in BE_REPLACEMENT_INTERVALS_YR:
        arc_be_results.append(
            (
                interval,
                pd.concat(
                    [
                        add_arc_be_replacement_demand(
                            arc_base[arc_base["scenario"] == scenario_name].copy(),
                            interval,
                        )
                        for scenario_name in SCENARIOS_TO_PLOT
                    ],
                    ignore_index=True,
                ),
            )
        )

    print_summary(base_df, arc_be_results)

    if SAVE_CSV:
        base_df.to_csv(OUT_CSV, index=False)
        print(f"[INFO] Wrote: {OUT_CSV}")

    plot_capacity_targets()

    for scenario_name in SCENARIOS_TO_PLOT:
        plot_reactor_comparison(
            base_df,
            scenario_name,
            "required_plants",
            "Required number of plants",
            "Required Plant Count",
        )

        plot_reactor_comparison(
            base_df,
            scenario_name,
            "new_builds",
            "New builds per year",
            "Annual New Builds",
        )

        plot_reactor_comparison(
            base_df,
            scenario_name,
            "rebco_annual_demand_km",
            "Annual REBCO demand (km)",
            "Annual REBCO Demand",
        )

        plot_lithium_actual_with_hammir_equiv(
            base_df,
            scenario_name,
        )

        plot_wb_nb3sn_dual_axis(
            base_df,
            scenario_name,
        )

    plot_arc_be_demand(arc_be_results)


if __name__ == "__main__":
    main()
