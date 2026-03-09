## 1. Serial Version

### Compile (run once from the folder)

```bash
gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
```

### Run

```bash
# Default: 1000 users, 1000 items
./serial_rec

# Custom sizes: ./serial_rec [num_users] [num_items]
./serial_rec 500 300
./serial_rec 2000 1500
```

---

## 2. OpenMP Shared Memory Version

### Compile (run once from the folder)

```bash
gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm
```

### Run – thread count via `OMP_NUM_THREADS`

```bash
# Default sizes (1000 users, 1000 items) – vary threads only
./openmp_rec
OMP_NUM_THREADS=2  ./openmp_rec
OMP_NUM_THREADS=4  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec

# Custom sizes: ./openmp_rec [num_users] [num_items]
OMP_NUM_THREADS=4  ./openmp_rec 500 300
OMP_NUM_THREADS=8  ./openmp_rec 2000 1500
```

---

## 3. Speedup Benchmark (for the report)

```bash
# Compile both
gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm

# Run and note "Total (sim+pred)" time from each output
./serial_rec

OMP_NUM_THREADS=1  ./openmp_rec
OMP_NUM_THREADS=2  ./openmp_rec
OMP_NUM_THREADS=4  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec
```