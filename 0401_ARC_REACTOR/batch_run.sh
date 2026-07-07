#!/bin/bash
# ============================================================
# Cyclus batch runner for NOAK 2D sweep
# ============================================================

set -e

XML_DIR="XML_CASES"
RUN_DIR="RUNS"

mkdir -p "${RUN_DIR}"

for xml in ${XML_DIR}/*.xml; do

    fname=$(basename "${xml}")
    tag="${fname%.xml}"

    echo "============================================"
    echo " Running NOAK case: ${tag}"
    echo "============================================"

    case_dir="${RUN_DIR}/${tag}"
    mkdir -p "${case_dir}"

    cp "${xml}" "${case_dir}/${fname}"

    pushd "${case_dir}" > /dev/null

    cyclus -o "out_${tag}.sqlite" "${fname}" \
        > "cyclus_${tag}.log" 2>&1

    popd > /dev/null

done

echo "============================================"
echo " All NOAK batch runs completed."
echo "============================================"