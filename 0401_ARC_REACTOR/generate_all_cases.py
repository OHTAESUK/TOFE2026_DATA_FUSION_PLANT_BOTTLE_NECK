#!/usr/bin/env python3
# ============================================================
# Generate 2D NOAK parameter sweep
# Be:   1–10 t/month
# CAGR: 0–10 %
# ============================================================

import os
import subprocess

XML_DIR = "XML_CASES"
os.makedirs(XML_DIR, exist_ok=True)

BE_RANGE = range(1, 11)      # 1–10 t/month
CAGR_RANGE = range(0, 11)    # 0–10 %

for be in BE_RANGE:
    for cagr in CAGR_RANGE:

        print(f"Generating Be={be} t/mo, CAGR={cagr}%")

        # Call your existing generator
        subprocess.run([
            "python",
            "NOAK_INPUT_GENERATOR.py"
        ], env={
            **os.environ,
            "INITIAL_BE_OVERRIDE": str(be * 1000),
            "CAGR_OVERRIDE": str(cagr / 100.0)
        })

        # Move generated XML
        fname = f"ARC_NOAK_Be{be}_CAGR{cagr}_FLiBeINF.xml"
        if os.path.exists(fname):
            os.rename(fname, os.path.join(XML_DIR, fname))

print("\n[INFO] All XML cases generated.\n")