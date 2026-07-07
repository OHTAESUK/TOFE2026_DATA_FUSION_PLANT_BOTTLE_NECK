#!/usr/bin/env python3
"""
Generate Cyclus input for:

Be-limited fleet deployment with annual CAGR growth.

Builder capacity is automatically sized to avoid artificial
deployment bottlenecks.

Output file name:
ARC_NOAK_Be{X}_CAGR{Y}_FLiBeINF.xml
"""

import os
import math

# ============================================================
# USER SETTINGS
# ============================================================

DURATION = 600
REPLACEMENT_INTERVAL = 12
FLIBE_THROUGHPUT = 1e9

# Receive sweep parameters
INITIAL_BE = float(os.getenv("INITIAL_BE_OVERRIDE", 10000))
CAGR = float(os.getenv("CAGR_OVERRIDE", 0.0))

# INITIAL_BE = 10_000
# CAGR = 0.10

# ============================================================
# CONSTANTS
# ============================================================

FLIBE_PER_ARC = 958000
BE_FRACTION = 0.0926
BE_PER_ARC = FLIBE_PER_ARC * BE_FRACTION

BUILDER_MARGIN = 1.2

# ============================================================
# AUTO FILE NAME
# ============================================================

be_tons = int(INITIAL_BE / 1000)
cagr_percent = int(CAGR * 100)

OUTPUT_FILE = f"ARC_NOAK_Be{be_tons}_CAGR{cagr_percent}_FLiBeINF.xml"

# ============================================================
# DERIVED
# ============================================================

n_years = DURATION // REPLACEMENT_INTERVAL

max_be_supply = INITIAL_BE * (1 + CAGR) ** (n_years - 1)

max_build_rate = max_be_supply / BE_PER_ARC

N_BUILDERS = max(1, math.ceil(max_build_rate * BUILDER_MARGIN) + 1)

print("--------------------------------------------------")
print(f"INITIAL_BE = {INITIAL_BE/1000:.1f} t/month")
print(f"CAGR = {CAGR*100:.1f} %")
print(f"Max Be supply = {max_be_supply:,.0f} kg/month")
print(f"Be per ARC = {BE_PER_ARC:,.0f} kg")
print(f"Theoretical max builds/month = {max_build_rate:.2f}")
print(f"Allocated builders = {N_BUILDERS}")
print("--------------------------------------------------")

# ============================================================
# FLIBE BUFFER SIZING
# ============================================================

BUFFER_MARGIN = 2.0

FLIBE_BUF_SIZE = int(FLIBE_PER_ARC * N_BUILDERS * BUFFER_MARGIN)

BE_BUF_SIZE = int(FLIBE_BUF_SIZE * 0.0926)
LI_BUF_SIZE = int(FLIBE_BUF_SIZE * 0.1384)
F_BUF_SIZE  = int(FLIBE_BUF_SIZE * 0.769)

print(f"FLiBe buffer size = {FLIBE_BUF_SIZE:,}")

# ============================================================
# START WRITING XML
# ============================================================

with open(OUTPUT_FILE, "w") as f:

    f.write("<simulation>\n\n")
    f.write("  <!-- Be-limited CAGR replacement model -->\n\n")

    # CONTROL
    f.write("  <control>\n")
    f.write(f"    <duration>{DURATION}</duration>\n")
    f.write("    <startyear>2000</startyear>\n")
    f.write("    <startmonth>1</startmonth>\n")
    f.write("    <decay>never</decay>\n")
    f.write("  </control>\n\n")

    # ARCHETYPES
    f.write("  <archetypes>\n")
    for lib, name in [
        ("cycamore", "Source"),
        ("cycamore", "Mixer"),
        ("cycamore", "Reactor"),
        ("cycamore", "Sink"),
        ("agents", "NullRegion"),
        ("cycamore", "DeployInst"),
    ]:
        f.write(f"    <spec><lib>{lib}</lib><name>{name}</name></spec>\n")
    f.write("  </archetypes>\n\n")

    # COMMODITIES
    for commod in ["Be_raw", "Li_raw", "F_raw", "FLiBe_raw", "built_flag"]:
        f.write(
            f"  <commodity><name>{commod}</name>"
            "<solution_priority>1.0</solution_priority></commodity>\n"
        )
    f.write("\n")

    # ========================================================
    # BE SOURCES
    # ========================================================

    for y in range(n_years):

        throughput = round(INITIAL_BE * (1 + CAGR) ** y)

        f.write("  <facility>\n")
        f.write(f"    <name>BeSource_Y{y}</name>\n")
        f.write(f"    <lifetime>{REPLACEMENT_INTERVAL}</lifetime>\n")
        f.write("    <config>\n")
        f.write("      <Source>\n")
        f.write("        <outcommod>Be_raw</outcommod>\n")
        f.write(f"        <throughput>{throughput}</throughput>\n")
        f.write("      </Source>\n")
        f.write("    </config>\n")
        f.write("  </facility>\n\n")

    # ========================================================
    # Li / F SOURCES
    # ========================================================

    for name in ["LiSource", "FSource"]:

        f.write("  <facility>\n")
        f.write(f"    <name>{name}</name>\n")
        f.write("    <config>\n")
        f.write("      <Source>\n")
        f.write(f"        <outcommod>{name.replace('Source','_raw')}</outcommod>\n")
        f.write("        <throughput>1000000000</throughput>\n")
        f.write("      </Source>\n")
        f.write("    </config>\n")
        f.write("  </facility>\n\n")

    # ========================================================
    # FLIBE FAB
    # ========================================================

    f.write("  <facility>\n")
    f.write("    <name>FLiBeFab</name>\n")
    f.write("    <config>\n")
    f.write("      <Mixer>\n")
    f.write("        <in_streams>\n")

    for ratio, buf, commod in [
        (0.0926, BE_BUF_SIZE, "Be_raw"),
        (0.1384, LI_BUF_SIZE, "Li_raw"),
        (0.7690, F_BUF_SIZE, "F_raw"),
    ]:

        f.write("          <stream>\n")
        f.write("            <info>\n")
        f.write(f"              <mixing_ratio>{ratio}</mixing_ratio>\n")
        f.write(f"              <buf_size>{buf}</buf_size>\n")
        f.write("            </info>\n")
        f.write("            <commodities>\n")
        f.write(
            f"              <item><commodity>{commod}</commodity><pref>1.0</pref></item>\n"
        )
        f.write("            </commodities>\n")
        f.write("          </stream>\n")

    f.write("        </in_streams>\n")
    f.write("        <out_commod>FLiBe_raw</out_commod>\n")
    f.write(f"        <out_buf_size>{int(FLIBE_BUF_SIZE)}</out_buf_size>\n")
    f.write(f"        <throughput>{int(FLIBE_THROUGHPUT)}</throughput>\n")
    f.write("      </Mixer>\n")
    f.write("    </config>\n")
    f.write("  </facility>\n\n")

    # ========================================================
    # ARC BUILDERS (AUTO)
    # ========================================================

    for i in range(N_BUILDERS):

        f.write("  <facility>\n")
        f.write(f"    <name>ARC_Builder_{i}</name>\n")
        f.write("    <config>\n")
        f.write("      <Reactor>\n")
        f.write("        <fuel_incommods><val>FLiBe_raw</val></fuel_incommods>\n")
        f.write("        <fuel_inrecipes><val>dummy_fresh</val></fuel_inrecipes>\n")
        f.write("        <fuel_outcommods><val>built_flag</val></fuel_outcommods>\n")
        f.write("        <fuel_outrecipes><val>dummy_spent</val></fuel_outrecipes>\n")
        f.write("        <assem_size>958000</assem_size>\n")
        f.write("        <n_assem_core>1</n_assem_core>\n")
        f.write("        <n_assem_batch>1</n_assem_batch>\n")
        f.write("        <n_assem_spent>1</n_assem_spent>\n")
        f.write("        <cycle_time>1</cycle_time>\n")
        f.write("        <refuel_time>0</refuel_time>\n")
        f.write("        <power_cap>1</power_cap>\n")
        f.write("      </Reactor>\n")
        f.write("    </config>\n")
        f.write("  </facility>\n\n")

    # ========================================================
    # FLAG SINK
    # ========================================================

    f.write("  <facility>\n")
    f.write("    <name>FlagSink</name>\n")
    f.write("    <config>\n")
    f.write("      <Sink>\n")
    f.write("        <in_commods><val>built_flag</val></in_commods>\n")
    f.write("        <capacity>1e12</capacity>\n")
    f.write("      </Sink>\n")
    f.write("    </config>\n")
    f.write("  </facility>\n\n")

    # ========================================================
    # DEPLOYMENT
    # ========================================================

    f.write("  <region>\n")
    f.write("    <name>world</name>\n")
    f.write("    <config><NullRegion/></config>\n")
    f.write("    <institution>\n")
    f.write("      <name>inst</name>\n")
    f.write("      <config>\n")
    f.write("        <DeployInst>\n")

    f.write("          <prototypes>\n")

    for y in range(n_years):
        f.write(f"            <val>BeSource_Y{y}</val>\n")

    for name in ["LiSource", "FSource", "FLiBeFab"]:
        f.write(f"            <val>{name}</val>\n")

    for i in range(N_BUILDERS):
        f.write(f"            <val>ARC_Builder_{i}</val>\n")

    f.write("            <val>FlagSink</val>\n")

    f.write("          </prototypes>\n")

    f.write("          <build_times>\n")

    for y in range(n_years):
        f.write(f"            <val>{1 + y * REPLACEMENT_INTERVAL}</val>\n")

    static_facilities = 3 + N_BUILDERS + 1

    for _ in range(static_facilities):
        f.write("            <val>1</val>\n")

    f.write("          </build_times>\n")

    f.write("          <n_build>\n")

    for _ in range(n_years + static_facilities):
        f.write("            <val>1</val>\n")

    f.write("          </n_build>\n")

    f.write("        </DeployInst>\n")
    f.write("      </config>\n")
    f.write("    </institution>\n")
    f.write("  </region>\n\n")

    # ========================================================
    # RECIPES
    # ========================================================

    for name in ["dummy_fresh", "dummy_spent"]:

        f.write("  <recipe>\n")
        f.write(f"    <name>{name}</name>\n")
        f.write("    <basis>mass</basis>\n")
        f.write("    <nuclide><id>26056</id><comp>1.0</comp></nuclide>\n")
        f.write("  </recipe>\n")

    f.write("</simulation>\n")

print(f"\nGenerated: {OUTPUT_FILE}\n")