#!/bin/bash
# =============================================================================
# run_benchmarks.sh
# Compile all versions, run them, and AUTO-FILL:
#   results/timing_raw.txt      — full terminal output of every run
#   results/speedup_table.csv   — parsed timings + computed speedup & efficiency
#   results/mae_comparison.txt  — MAE and sim-matrix checksum per version
#
# Usage (from the project root directory):
#   bash results/run_benchmarks.sh
#
# Requirements:
#   gcc, mpicc, mpirun must be on PATH.
#   For CUDA runs: uncomment the cuda block and ensure nvcc is available.
#   On an HPC cluster: load modules first, e.g.
#     module load gcc openmpi cuda
# =============================================================================

set -e   # exit immediately if any command fails

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # project root (one level up from results/)
RES="$ROOT/results"
RAW="$RES/timing_raw.txt"
CSV="$RES/speedup_table.csv"
MAE="$RES/mae_comparison.txt"

SIZE="1000 1000"   # benchmark problem size (N_USERS N_ITEMS)

# ── Helper: extract a timing value from one run's output ──────────────────
# Usage:  get_time "label" "$output_string"
# Looks for lines like:  [Timing] Similarity matrix  : 4.5678 s
get_time() {
    local label="$1"
    local text="$2"
    echo "$text" | grep "\[Timing\] $label" | grep -oP '[0-9]+\.[0-9]+' | head -1
}

get_mae() {
    local text="$1"
    echo "$text" | grep "\[Eval\]" | grep -oP '[0-9]+\.[0-9]+' | head -1
}

get_checksum() {
    local text="$1"
    echo "$text" | grep "\[Check\]" | grep -oP '[0-9]+\.[0-9]+' | head -1
}

# ── Compile ───────────────────────────────────────────────────────────────
echo "=== Compiling all versions ==="
gcc  -O2 -Wall           -o "$ROOT/serial_rec"   "$ROOT/serial/serial_recommender.c"              -lm
gcc  -O2 -Wall -fopenmp  -o "$ROOT/openmp_rec"   "$ROOT/openmp/openmp_recommender.c"              -lm
gcc  -O2 -Wall           -o "$ROOT/pthreads_rec" "$ROOT/pthreads/pthreads_recommender.c" -lpthread -lm
mpicc -O2 -Wall          -o "$ROOT/mpi_rec"      "$ROOT/mpi/mpi_recommender.c"                    -lm
mpicc -O2 -Wall -fopenmp -o "$ROOT/hybrid_rec"   "$ROOT/hybrid/hybrid_recommender.c"              -lm

# CUDA is optional: only build it when the toolkit (nvcc) is present, so this
# script still runs end-to-end on a CPU-only machine.
HAVE_CUDA=0
if command -v nvcc >/dev/null 2>&1; then
    nvcc -O2 -arch=sm_86 -o "$ROOT/cuda_rec" "$ROOT/cuda/cuda_recommender.cu" -lm \
        && HAVE_CUDA=1 || { echo "  (nvcc failed — skipping CUDA)"; HAVE_CUDA=0; }
else
    echo "  (nvcc not found — skipping CUDA build/run)"
fi
echo "Compile done."
echo ""

# ── Initialise output files ───────────────────────────────────────────────
echo "HPC Group 11 — Benchmark Run  $(date)" > "$RAW"
echo "Problem size: $SIZE  |  SEED=42  |  TOP_K=20"  >> "$RAW"
echo "=========================================="      >> "$RAW"

# CSV header
cat > "$CSV" <<'EOF'
Version,Workers,Sim_Time_s,Pred_Time_s,Total_s,Speedup,Efficiency
EOF

# MAE header
cat > "$MAE" <<'EOF'
# MAE Correctness Comparison  (1000 x 1000, SEED=42)
# All versions must match Serial MAE within +/-0.001
#
Version         | MAE    | Matches Serial? | Sim Checksum
----------------|--------|-----------------|------------------
EOF

# ── Run a version and extract metrics ─────────────────────────────────────
# Globals set by run_and_record():
SERIAL_TOTAL=""   # filled by serial run; used to compute all speedups

run_and_record() {
    local label="$1"    # human label for CSV (e.g. "OpenMP")
    local workers="$2"  # worker count for CSV (e.g. 4)
    local cmd="$3"      # full command to run

    echo "--- $label ($workers workers) ---"
    echo ""
    echo "=== $label ($workers workers) ===" >> "$RAW"

    # Run the command; capture output AND show it on screen
    local out
    out=$(eval "$cmd" 2>&1)
    echo "$out" | tee -a "$RAW"
    echo "" >> "$RAW"

    # Parse the three key values from the output
    local sim  pred  total  mae  csum
    sim=$(get_time  "Similarity matrix" "$out")
    pred=$(get_time "Prediction phase"  "$out")
    total=$(get_time "Total (sim+pred)" "$out")
    mae=$(get_mae      "$out")
    csum=$(get_checksum "$out")

    # Set serial baseline on first run
    if [ "$label" = "Serial" ]; then
        SERIAL_TOTAL="$total"
    fi

    # Compute speedup and efficiency (requires bc for floating-point arithmetic)
    local speedup efficiency
    if [ -n "$SERIAL_TOTAL" ] && [ -n "$total" ] && command -v bc &>/dev/null; then
        speedup=$(echo "scale=4; $SERIAL_TOTAL / $total" | bc)
        efficiency=$(echo "scale=4; $speedup / $workers" | bc)
    else
        speedup="N/A"
        efficiency="N/A"
    fi

    # Append row to CSV
    echo "$label,$workers,${sim:-N/A},${pred:-N/A},${total:-N/A},$speedup,$efficiency" >> "$CSV"

    # Append row to MAE file
    # Check if MAE matches serial (within 0.001)
    local match="YES"
    if [ "$label" = "Serial" ]; then
        match="baseline"
        SERIAL_MAE="$mae"
    elif [ -n "$SERIAL_MAE" ] && [ -n "$mae" ]; then
        match=$(awk -v a="$mae" -v b="$SERIAL_MAE" 'BEGIN{
            d = a - b; if (d < 0) d = -d;
            if (d <= 0.001) printf "YES";
            else            printf "NO - diff=%.4f", d;
        }')
    fi

    printf "%-16s| %-7s| %-16s| %s\n" \
        "$label" "${mae:-N/A}" "$match" "${csum:-N/A}" >> "$MAE"

    echo ""
}

# ── 1. Serial baseline ────────────────────────────────────────────────────
run_and_record "Serial" "1" \
    "\"$ROOT/serial_rec\" $SIZE"

# ── 2. OpenMP ─────────────────────────────────────────────────────────────
for T in 1 2 4 8; do
    run_and_record "OpenMP" "$T" \
        "OMP_NUM_THREADS=$T \"$ROOT/openmp_rec\" $SIZE"
done

# ── 3. Pthreads ───────────────────────────────────────────────────────────
for T in 1 2 4 8; do
    run_and_record "Pthreads" "$T" \
        "\"$ROOT/pthreads_rec\" $SIZE $T"
done

# ── 4. MPI ────────────────────────────────────────────────────────────────
for P in 1 2 4 8; do
    run_and_record "MPI" "$P" \
        "mpirun -np $P \"$ROOT/mpi_rec\" $SIZE"
done

# ── 5. CUDA (only if the toolkit was found at compile time) ───────────────
if [ "$HAVE_CUDA" = 1 ]; then
    run_and_record "CUDA" "GPU" \
        "\"$ROOT/cuda_rec\" $SIZE"
else
    echo "--- CUDA skipped (no nvcc) ---"
    echo ""
fi

# ── 6. Hybrid ─────────────────────────────────────────────────────────────
for CONFIG in "2 4" "4 2" "8 1" "1 8"; do
    P=$(echo $CONFIG | awk '{print $1}')
    T=$(echo $CONFIG | awk '{print $2}')
    WORKERS=$((P * T))
    run_and_record "Hybrid_${P}x${T}" "$WORKERS" \
        "mpirun -np $P env OMP_NUM_THREADS=$T \"$ROOT/hybrid_rec\" $SIZE"
done

# ── 7. Scalability sweep across problem sizes (Analysis Report §4.7) ───────
# Strong-scaling above varied the worker count at a fixed 1000x1000 problem.
# Here we vary the PROBLEM SIZE at fixed best configs (Serial, OpenMP 8T,
# MPI 8P, CUDA) to show how each model sustains its speedup as N grows.
SCALE_CSV="$RES/scaling_summary.csv"
echo "=== Scalability sweep across problem sizes ==="
cat > "$SCALE_CSV" <<'EOF'
Size,Serial_s,OpenMP8_s,OpenMP8_speedup,MPI8_s,MPI8_speedup,CUDA_s,CUDA_speedup
EOF

for SZ in "500 500" "1000 1000" "2000 2000"; do
    label=$(echo "$SZ" | tr ' ' 'x')
    echo "--- scaling size $label ---"
    echo "=== Scaling size $label ===" >> "$RAW"

    s_out=$(eval "\"$ROOT/serial_rec\" $SZ" 2>&1);                    echo "$s_out" >> "$RAW"
    o_out=$(eval "OMP_NUM_THREADS=8 \"$ROOT/openmp_rec\" $SZ" 2>&1);  echo "$o_out" >> "$RAW"
    m_out=$(eval "mpirun -np 8 \"$ROOT/mpi_rec\" $SZ" 2>&1);          echo "$m_out" >> "$RAW"
    c_out=""
    if [ "$HAVE_CUDA" = 1 ]; then
        c_out=$(eval "\"$ROOT/cuda_rec\" $SZ" 2>&1);                  echo "$c_out" >> "$RAW"
    fi

    s_t=$(get_time "Total (sim+pred)" "$s_out")
    o_t=$(get_time "Total (sim+pred)" "$o_out")
    m_t=$(get_time "Total (sim+pred)" "$m_out")
    c_t=$(get_time "Total (sim+pred)" "$c_out")

    osp="N/A"; msp="N/A"; csp="N/A"
    if command -v bc &>/dev/null && [ -n "$s_t" ]; then
        [ -n "$o_t" ] && osp=$(echo "scale=2; $s_t / $o_t" | bc)
        [ -n "$m_t" ] && msp=$(echo "scale=2; $s_t / $m_t" | bc)
        [ -n "$c_t" ] && csp=$(echo "scale=2; $s_t / $c_t" | bc)
    fi
    echo "$label,${s_t:-N/A},${o_t:-N/A},$osp,${m_t:-N/A},$msp,${c_t:-N/A},$csp" >> "$SCALE_CSV"
done
echo ""

# ── Done ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Results written to:"
echo "  $RAW         (full raw output)"
echo "  $CSV         (speedup table, 1000x1000 worker sweep)"
echo "  $MAE         (MAE correctness)"
echo "  $SCALE_CSV   (problem-size scalability sweep)"
echo "=========================================="
