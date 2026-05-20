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

---

## 3. MPI Distributed Memory Version

### Compile (run once from the folder)

```bash
mpicc -O2 -Wall -o mpi_rec mpi/mpi_recommender.c -lm
```

### Run – process count via `mpirun -np`

```bash
# Default sizes (1000 users, 1000 items) – vary process count only
mpirun -np 2 ./mpi_rec
mpirun -np 4 ./mpi_rec
mpirun -np 8 ./mpi_rec

# Custom sizes: mpirun -np <num_procs> ./mpi_rec [num_users] [num_items]
mpirun -np 4 ./mpi_rec 500 300
mpirun -np 2 ./mpi_rec 2000 1500
mpirun -np 4 ./mpi_rec 2000 1500
```

### Key Features
- **Distributed similarity computation**: Each process computes its assigned block of user similarity matrix rows
- **Data broadcasting**: Rating matrix distributed from rank 0 to all processes
- **Allgatherv**: Efficient gathering of similarity rows and predictions across all processes
- **Scalability**: Linear scaling with number of MPI processes (up to the number of users)

---

## 4. Speedup Benchmark (for the report)

```bash
# Compile all three versions
gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm
mpicc -O2 -Wall -o mpi_rec mpi/mpi_recommender.c -lm

# Serial baseline
./serial_rec

# OpenMP scaling (1 thread = baseline)
OMP_NUM_THREADS=1  ./openmp_rec
OMP_NUM_THREADS=2  ./openmp_rec
OMP_NUM_THREADS=4  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec

mpirun -np 1 ./mpi_rec
mpirun -np 2 ./mpi_rec
mpirun -np 4 ./mpi_rec
mpirun -np 8 ./mpi_rec
```

### Performance Analysis
- Compare **execution times** across serial, OpenMP, and MPI versions
- Calculate **speedup** = Serial_Time / Parallel_Time
- Analyze **efficiency** = Speedup / Number_of_Processors
- Observe **strong scaling** (fixed problem size, increase processors)
- Observe **weak scaling** (increase both problem size and processors proportionally)
