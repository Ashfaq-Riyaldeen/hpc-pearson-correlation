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

## 3. CUDA GPU-Accelerated Version

### Requirements
- NVIDIA GPU with CUDA compute capability 3.0+
- CUDA Toolkit installed (`nvcc --version`)

### Compile (run once from the folder)

```bash
nvcc -O2 -o cuda_rec cuda/cuda_recommender.cu -lm
```

### Run

```bash
# Default: 1000 users, 1000 items
./cuda_rec

# Custom sizes: ./cuda_rec [num_users] [num_items]
./cuda_rec 500 300
./cuda_rec 2000 1500
```

### Key Features
- **GPU-accelerated similarity**: All Pearson correlations computed on GPU
- **GPU-accelerated predictions**: Top-K neighbor selection and predictions on GPU
- **Efficient transfers**: Minimal host-device data movement
- **Scalability**: Leverages massive GPU parallelism for large problem sizes

---

## 4. Hybrid CUDA+MPI Distributed GPU Version

### Requirements
- Multiple nodes with NVIDIA GPUs
- MPI installed (OpenMPI or MPICH)
- CUDA Toolkit installed on all nodes

### Compile (run once from the folder)

```bash
# Using OpenMPI:
mpicxx -O2 -o cuda_mpi_rec cuda/cuda_mpi_recommender.cu -lm -L/usr/local/cuda/lib64 -lcudart

# Or using MPICH:
mpicc -O2 -o cuda_mpi_rec cuda/cuda_mpi_recommender.cu -lm -L/usr/local/cuda/lib64 -lcudart
```

### Run – process count and GPU assignment

```bash
# Default sizes (1000 users, 1000 items) – vary process count only
mpirun -np 2 ./cuda_mpi_rec
mpirun -np 4 ./cuda_mpi_rec
mpirun -np 8 ./cuda_mpi_rec

# Custom sizes: mpirun -np <num_procs> ./cuda_mpi_rec [num_users] [num_items]
mpirun -np 2 ./cuda_mpi_rec 500 300
mpirun -np 4 ./cuda_mpi_rec 2000 1500
mpirun -np 8 ./cuda_mpi_rec 5000 3000
```

### Key Features
- **Distributed GPU computation**: Each MPI process controls one GPU
- **Local similarity blocks**: Each rank computes assigned user similarities on its GPU
- **GPU-accelerated predictions**: Predictions computed on GPU before gathering
- **Allgatherv**: Efficient inter-process communication of similarity matrix and predictions
- **Scalability**: Combines multi-GPU with multi-node parallelism

---

## 5. Speedup Benchmark (for the report)

```bash
# Compile all versions
gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm
mpicc -O2 -Wall -o mpi_rec mpi/mpi_recommender.c -lm
nvcc -O2 -o cuda_rec cuda/cuda_recommender.cu -lm
mpicxx -O2 -o cuda_mpi_rec cuda/cuda_mpi_recommender.cu -lm -L/usr/local/cuda/lib64 -lcudart

# Serial baseline
./serial_rec

# OpenMP scaling (1 thread = baseline)
OMP_NUM_THREADS=1  ./openmp_rec
OMP_NUM_THREADS=2  ./openmp_rec
OMP_NUM_THREADS=4  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec

# MPI scaling
mpirun -np 1 ./mpi_rec
mpirun -np 2 ./mpi_rec
mpirun -np 4 ./mpi_rec
mpirun -np 8 ./mpi_rec

# GPU acceleration
./cuda_rec

# Hybrid GPU+MPI
mpirun -np 2 ./cuda_mpi_rec
mpirun -np 4 ./cuda_mpi_rec
```

### Performance Analysis
- **Serial baseline**: Captures inherent algorithm cost
- **OpenMP scaling**: Strong scaling with shared-memory parallelism
- **MPI scaling**: Distributed-memory efficiency (communication overhead vs. computation)
- **GPU acceleration**: Compare CUDA vs. serial (order of magnitude difference expected)
- **Hybrid GPU+MPI**: Multi-GPU distributed speedup with communication trade-offs
- **Speedup calculation**: Serial_Time / Parallel_Time
- **Efficiency**: Speedup / Number_of_Processors
- **Strong scaling** (fixed 1000×1000): Increase processes to see efficiency drop
- **Weak scaling** (proportional increase): e.g., 4 procs with 4000 users, 8 procs with 8000 users