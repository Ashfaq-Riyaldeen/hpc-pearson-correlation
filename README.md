# Parallel Pearson-Correlation Recommender System

**EC7207 / EE7218 — High Performance Computing · Group 11**

A user-based collaborative-filtering recommender (Pearson-correlation user
similarity + top-K weighted prediction) implemented **six ways** so their
performance and accuracy can be compared: a **serial** baseline plus five
parallel versions — **OpenMP**, **POSIX Threads**, **MPI**, **CUDA**, and a
**Hybrid MPI+OpenMP**. Every version runs the *same* algorithm on the *same*
data (`SEED=42`), so only the runtime differs — and all of them produce an
identical MAE, RMSE, and similarity-matrix checksum, which is how correctness
is verified.

- Group: EG/2021/4417 (Ashfaq M.R.M.) · EG/2021/4419 (Athanayaka K.A.L.G.) · EG/2021/4424 (Balasooriya J.M.)
- Reference platform: Linux (Pop!_OS 24.04), Intel Core i7-11800H (8 physical / 16 logical), NVIDIA GeForce RTX 3050 Laptop GPU (compute capability **8.6**).

---

## Repository structure

```
serial/    openmp/    pthreads/    mpi/    cuda/    hybrid/   # one *_recommender.{c,cu} per version
results/             # run_benchmarks.sh (canonical runner) + benchmark data (CSV / txt)
analysis_report/     # ANALYSIS_REPORT.md  +  analysis_report.tex / .pdf  (the report)
analysis_diagrams/   # drawio/ concept diagrams (+ png/), charts/, generate_charts.py
learning/            # per-technology learning material
Project Guideline.pdf   HPC_Group_11.pdf   # assignment brief + project proposal
README.md   CLAUDE.md
```

---

## Prerequisites

| Tool | Needed for | Notes |
|------|------------|-------|
| GCC (with `-fopenmp`) | Serial, OpenMP, Pthreads | `build-essential` on Debian/Ubuntu |
| An MPI implementation | MPI, Hybrid | Open MPI (`mpicc`/`mpirun`) on Linux, or MS-MPI (`mpiexec`) on Windows |
| NVIDIA CUDA Toolkit (`nvcc`) + NVIDIA GPU | CUDA only | Match `-arch` to your GPU (see §4). The other five versions need **no** GPU |
| `bc`, `awk`, `grep`, `bash` | `results/run_benchmarks.sh` | Standard on Linux |
| Python 3 + `matplotlib`, `numpy` | Regenerating charts | `analysis_diagrams/generate_charts.py` |
| LaTeX (`pdflatex`, e.g. TeX Live / MiKTeX) | Rebuilding the report PDF | `analysis_report/analysis_report.tex` |

---

## Quick start — run the whole benchmark

The canonical runner compiles every version, sweeps worker counts (1/2/4/8)
and problem sizes (500/1000/2000), and writes the results files:

```bash
bash results/run_benchmarks.sh        # run from the project root
```

Outputs (in `results/`):
- `timing_raw.txt` — full output of every run
- `speedup_table.csv` — parsed timings + computed speedup & efficiency
- `mae_comparison.txt` — MAE / RMSE / checksum per version
- `scaling_summary.csv` — problem-size scalability sweep

CUDA is built/run only if `nvcc` is present, so the script also works on a
CPU-only machine. A secondary Windows CPU-only runner exists at
`results/run_win_bench.sh` (writes to `timing_raw_windows.txt`).

To regenerate the report charts from the data, then rebuild the PDF:

```bash
python analysis_diagrams/generate_charts.py            # reads results/*.csv → charts/*.png
cd analysis_report && pdflatex analysis_report.tex && pdflatex analysis_report.tex
```

---

## Build & run each version individually

All compile commands are run from the project root.

### 1. Serial (baseline)

```bash
gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
```

```bash
./serial_rec                 # default 1000 users, 1000 items
./serial_rec 500 300         # custom: ./serial_rec [num_users] [num_items]
./serial_rec 2000 1500
```

### 2. OpenMP (shared memory)

```bash
gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm
```

Thread count is set with `OMP_NUM_THREADS`:

```bash
./openmp_rec                          # default thread count
OMP_NUM_THREADS=2  ./openmp_rec
OMP_NUM_THREADS=4  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec 2000 1500   # custom size
```

### 3. Pthreads (shared memory)

```bash
gcc -O2 -Wall -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm
```

Thread count is the **third** argument:

```bash
./pthreads_rec                  # default 1000 x 1000, 4 threads
./pthreads_rec 1000 1000 1
./pthreads_rec 1000 1000 2
./pthreads_rec 1000 1000 4
./pthreads_rec 1000 1000 8
./pthreads_rec 2000 1500 8      # custom size + threads
```

### 4. CUDA (GPU)

```bash
nvcc -O2 -arch=sm_86 -o cuda_rec cuda/cuda_recommender.cu -lm
```

> `sm_86` matches this project's GPU (RTX 3050, compute capability 8.6).
> Replace it with your GPU's compute capability:
> | `-arch` | GPU family |
> |---------|------------|
> | `sm_60` | Pascal (GTX 10xx) |
> | `sm_70` | Volta (V100) |
> | `sm_75` | Turing (RTX 20xx, GTX 16xx) |
> | `sm_80` | Ampere (A100) |
> | `sm_86` | Ampere (RTX 30xx) |
> | `sm_89` | Ada Lovelace (RTX 40xx) |
>
> Detect it with: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`

```bash
./cuda_rec                # default 1000 x 1000
./cuda_rec 2000 1500      # custom size
```

What runs on the GPU:

| Phase | Kernel | Parallelism |
|-------|--------|-------------|
| User means | `kernel_user_means` | 1 thread per user (1-D grid) |
| Similarity matrix | `kernel_similarity` | 1 thread per (u,v) pair (2-D grid) |
| Predictions | `kernel_predictions` | 1 thread per (user, item) pair (2-D grid) |

Kernel time is measured with CUDA events; host↔device transfer time is reported
separately.

### 5. MPI (distributed memory)

```bash
mpicc -O2 -Wall -o mpi_rec mpi/mpi_recommender.c -lm
```

Process count is set with `-np`:

```bash
mpirun -np 1  ./mpi_rec
mpirun -np 2  ./mpi_rec
mpirun -np 4  ./mpi_rec
mpirun -np 8  ./mpi_rec
mpirun -np 4  ./mpi_rec 2000 1500    # custom size
```

> If `mpirun` is unavailable, try `mpiexec -n 4 ./mpi_rec`. On a cluster you may
> need to load a module first: `module load openmpi` (or `mpich`).

### 6. Hybrid MPI + OpenMP

Combines both models: **MPI** divides users across processes (coarse grain,
private address space per rank); **OpenMP** threads within each rank cooperate
over that rank's rows in shared memory. With **P** ranks × **T** threads you get
**P × T** workers while sending only **P** messages per collective.

```bash
mpicc -O2 -Wall -fopenmp -o hybrid_rec hybrid/hybrid_recommender.c -lm
```

Set MPI processes with `-np` and OpenMP threads with `OMP_NUM_THREADS`:

```bash
mpirun -np 2 env OMP_NUM_THREADS=4 ./hybrid_rec     # 2 × 4 = 8 workers
mpirun -np 4 env OMP_NUM_THREADS=2 ./hybrid_rec     # 4 × 2 = 8 workers
mpirun -np 1 env OMP_NUM_THREADS=8 ./hybrid_rec     # pure-OpenMP-like
mpirun -np 2 env OMP_NUM_THREADS=4 ./hybrid_rec 2000 1500
```

> On some systems use `-x OMP_NUM_THREADS=4` (Open MPI) or
> `-genv OMP_NUM_THREADS 4` (Intel MPI) instead of `env`.

| Phase | MPI (between processes) | OpenMP (within a process) |
|-------|-------------------------|---------------------------|
| Data generation | each rank runs the same RNG | serial (data is identical) |
| User means | each rank computes its rows | `parallel for schedule(static)` |
| Similarity | each rank computes its rows | `parallel for schedule(dynamic,4)` |
| Gather | `MPI_Allgatherv` | — |
| Predictions | each rank predicts its users | `parallel` + private buffer |
| MAE / RMSE | `MPI_Reduce` | `reduction(+:...)` |

---

## Reproducing the speedup table by hand

`results/run_benchmarks.sh` does this automatically; the manual equivalent is:

```bash
# Compile all versions
gcc   -O2 -Wall            -o serial_rec   serial/serial_recommender.c              -lm
gcc   -O2 -Wall -fopenmp   -o openmp_rec   openmp/openmp_recommender.c              -lm
gcc   -O2 -Wall            -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm
mpicc -O2 -Wall            -o mpi_rec      mpi/mpi_recommender.c                    -lm
mpicc -O2 -Wall -fopenmp   -o hybrid_rec   hybrid/hybrid_recommender.c              -lm
nvcc  -O2 -arch=sm_86      -o cuda_rec     cuda/cuda_recommender.cu                 -lm

# Baseline + worker sweeps
./serial_rec 1000 1000
for T in 1 2 4 8; do OMP_NUM_THREADS=$T ./openmp_rec 1000 1000; done
for T in 1 2 4 8; do ./pthreads_rec 1000 1000 $T; done
for P in 1 2 4 8; do mpirun -np $P ./mpi_rec 1000 1000; done
./cuda_rec 1000 1000
for C in "2 4" "4 2" "8 1" "1 8"; do set -- $C; mpirun -np $1 env OMP_NUM_THREADS=$2 ./hybrid_rec 1000 1000; done
```

Read the `[Timing] Total (sim+pred)` line from each run.
**Speedup = Serial_total / Parallel_total.**

---

## Results, report & diagrams

- **Report:** `analysis_report/analysis_report.pdf` (built from `analysis_report.tex`); a Markdown twin is at `analysis_report/ANALYSIS_REPORT.md`.
- **Concept diagrams:** `analysis_diagrams/drawio/` (editable `.drawio` + exported `png/`).
- **Performance charts:** `analysis_diagrams/charts/` (generated from `results/` by `generate_charts.py`).
- **Raw data:** `results/timing_raw.txt`, `speedup_table.csv`, `mae_comparison.txt`.

## Correctness

Every version (CPU and GPU) produces an identical **MAE = 1.2574**,
**RMSE = 1.4579**, and **similarity-matrix checksum = 942.387323** for the
default 1000×1000 / `SEED=42` input. Matching results across all versions is
the project's correctness check — any deviation would indicate a race condition
or a partitioning bug.
