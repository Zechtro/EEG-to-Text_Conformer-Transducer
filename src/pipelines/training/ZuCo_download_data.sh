#!/usr/bin/env bash
# ============================================================================
# ZuCo NR Dataset Downloader
# ============================================================================
# Downloads the preprocessed Normal Reading (NR) Matlab files for all subjects
# from ZuCo v1.0 (OSF: q3zws) and v2.0 (OSF: 2urht).
#
# Each file is ~200-500 MB. Total: ~10-15 GB for all 30 subjects.
# /workspace has ~149 GB free, so there is plenty of space.
#
# Usage:
#   bash ZuCo_download_data.sh           # download all subjects
#   bash ZuCo_download_data.sh v1        # only v1 (12 subjects)
#   bash ZuCo_download_data.sh v2        # only v2 (18 subjects)
#   bash ZuCo_download_data.sh ZGW v1    # only one subject (for quick test)
# ============================================================================

OSF=/venv/main/bin/osf
V1_DIR=/workspace/zuco_data/v1/NR
V2_DIR=/workspace/zuco_data/v2/NR
mkdir -p "$V1_DIR" "$V2_DIR"

# ZuCo v1.0 subjects (12) — NR task is "task2 - NR" on OSF q3zws
V1_SUBJECTS=(ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH)

# ZuCo v2.0 subjects (18) — NR task is "task1 - NR" on OSF 2urht
# Note: YDR appears in OSF instead of YDG in some versions; both are included
V2_SUBJECTS=(YAC YAG YAK YDG YDR YFR YFS YHS YIS YLS YMD YMS YRH YRK YRP YSD YSL YTL YTT)

download_v1() {
    local subj="$1"
    local dst="$V1_DIR/results${subj}_NR.mat"
    if [ -f "$dst" ]; then
        echo "[SKIP] $dst already exists"
        return
    fi
    echo "[v1] Downloading results${subj}_NR.mat ..."
    $OSF -p q3zws fetch "osfstorage/task2 - NR/Matlab files/results${subj}_NR.mat" "$dst" \
        && echo "  -> OK" || echo "  -> FAILED (subject $subj may not be in v1)"
}

download_v2() {
    local subj="$1"
    local dst="$V2_DIR/results${subj}_NR.mat"
    if [ -f "$dst" ]; then
        echo "[SKIP] $dst already exists"
        return
    fi
    echo "[v2] Downloading results${subj}_NR.mat ..."
    $OSF -p 2urht fetch "osfstorage/task1 - NR/Matlab files/results${subj}_NR.mat" "$dst" \
        && echo "  -> OK" || echo "  -> FAILED (subject $subj may not be in v2)"
}

# --- Argument handling ---
MODE="${1:-all}"   # all / v1 / v2 / <SUBJECT_ID>

if [[ "$MODE" == "v1" ]]; then
    echo "=== Downloading ZuCo v1.0 NR (12 subjects) ==="
    for s in "${V1_SUBJECTS[@]}"; do download_v1 "$s"; done

elif [[ "$MODE" == "v2" ]]; then
    echo "=== Downloading ZuCo v2.0 NR (18 subjects) ==="
    for s in "${V2_SUBJECTS[@]}"; do download_v2 "$s"; done

elif [[ "$MODE" == "all" ]]; then
    echo "=== Downloading ZuCo v1.0 NR (12 subjects) ==="
    for s in "${V1_SUBJECTS[@]}"; do download_v1 "$s"; done
    echo ""
    echo "=== Downloading ZuCo v2.0 NR (18 subjects) ==="
    for s in "${V2_SUBJECTS[@]}"; do download_v2 "$s"; done

else
    # Single subject: bash ZuCo_download_data.sh ZGW v1
    SUBJ="$1"
    VER="${2:-v1}"
    echo "=== Downloading single subject: $SUBJ ($VER) ==="
    if [[ "$VER" == "v1" ]]; then
        download_v1 "$SUBJ"
    else
        download_v2 "$SUBJ"
    fi
fi

echo ""
echo "=== Done. Files in: ==="
ls -lh "$V1_DIR/"*.mat 2>/dev/null || echo "  (no v1 files yet)"
ls -lh "$V2_DIR/"*.mat 2>/dev/null || echo "  (no v2 files yet)"
