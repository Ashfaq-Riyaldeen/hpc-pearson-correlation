/*
 * cuda_recommender.cu
 * Pearson Correlation Recommender – CUDA GPU Version
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WHAT IS CUDA?                                                          ║
 * ║                                                                         ║
 * ║  CUDA (Compute Unified Device Architecture) is NVIDIA's parallel       ║
 * ║  computing platform.  It lets you write C/C++ functions called         ║
 * ║  "kernels" that run on thousands of GPU threads simultaneously.        ║
 * ║                                                                         ║
 * ║  The key difference from CPU parallelism:                              ║
 * ║   ● A CPU has 4–64 POWERFUL cores, each with large caches and         ║
 * ║     branch-prediction hardware — great for complex, sequential logic.  ║
 * ║   ● A GPU has thousands of SIMPLE cores — great for data-parallel      ║
 * ║     work where the same operation is applied to many independent        ║
 * ║     elements at the same time.                                          ║
 * ║                                                                         ║
 * ║  Our similarity matrix has N_USERS² independent (u,v) pairs — each    ║
 * ║  pair's Pearson correlation is completely independent of every other.  ║
 * ║  This is a textbook GPU workload.                                       ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  GPU THREAD HIERARCHY  (most important concept to understand first)     ║
 * ║                                                                         ║
 * ║  CUDA organises threads in a three-level hierarchy:                    ║
 * ║                                                                         ║
 * ║  THREAD  — the smallest unit of execution.  Each thread runs the       ║
 * ║            kernel function independently and has its own registers      ║
 * ║            and a small private "local memory" (spill space on DRAM).  ║
 * ║                                                                         ║
 * ║  BLOCK   — a group of threads (up to 1024) that run on the SAME        ║
 * ║            Streaming Multiprocessor (SM).  Threads in a block can      ║
 * ║            communicate via fast SHARED MEMORY and synchronise with     ║
 * ║            __syncthreads().  Threads in different blocks CANNOT share  ║
 * ║            memory or synchronise directly.                              ║
 * ║                                                                         ║
 * ║  GRID    — the collection of ALL blocks launched for one kernel call.  ║
 * ║            The GPU scheduler distributes blocks across SMs.            ║
 * ║                                                                         ║
 * ║  Visual (2D grid used by the similarity kernel):                       ║
 * ║                                                                         ║
 * ║   GRID (gridDim.x × gridDim.y blocks)                                  ║
 * ║   ┌──────────────────────────────────────┐                             ║
 * ║   │ Block(0,0) │ Block(1,0) │ Block(2,0) │  ← blockIdx.y = 0          ║
 * ║   │ Block(0,1) │ Block(1,1) │ Block(2,1) │  ← blockIdx.y = 1          ║
 * ║   └──────────────────────────────────────┘                             ║
 * ║           │                                                             ║
 * ║           ▼  each block has blockDim.x × blockDim.y threads            ║
 * ║   ┌──────────────────────────────────────┐                             ║
 * ║   │ T(0,0) T(1,0) … T(15,0)             │                             ║
 * ║   │ T(0,1) T(1,1) … T(15,1)  ← 16×16   │                             ║
 * ║   │  …                        = 256 thr  │                             ║
 * ║   │ T(0,15)          T(15,15)            │                             ║
 * ║   └──────────────────────────────────────┘                             ║
 * ║                                                                         ║
 * ║  Each thread computes its unique (u,v) index as:                       ║
 * ║    u = blockIdx.x * blockDim.x + threadIdx.x                           ║
 * ║    v = blockIdx.y * blockDim.y + threadIdx.y                           ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  GPU MEMORY HIERARCHY  (performance-critical knowledge)                 ║
 * ║                                                                         ║
 * ║  From fastest to slowest:                                               ║
 * ║                                                                         ║
 * ║  REGISTERS    ← per-thread, sub-nanosecond, ~65536 per SM              ║
 * ║                  Used for local scalar variables and small arrays.      ║
 * ║                  The top_sim/top_rat/top_mu arrays in our prediction    ║
 * ║                  kernel live here (when they fit).                      ║
 * ║                                                                         ║
 * ║  SHARED MEM   ← per-block, ~1 cycle, ~48 KB per SM                    ║
 * ║                  Declared with __shared__.  Used for inter-thread       ║
 * ║                  cooperation within a block.  We don't use it in       ║
 * ║                  this file (our kernels are embarrassingly parallel).  ║
 * ║                                                                         ║
 * ║  L1 / L2      ← hardware-managed caches, transparent to the code.     ║
 * ║                                                                         ║
 * ║  GLOBAL MEM   ← GPU DRAM, ~200–400 GB/s, ~300–600 cycle latency.      ║
 * ║                  d_ratings, d_user_mean, d_sim, d_pred all live here.  ║
 * ║                  Accesses should be COALESCED: adjacent threads should  ║
 * ║                  access adjacent memory addresses so the GPU can        ║
 * ║                  serve all threads in a warp with one memory request.  ║
 * ║                                                                         ║
 * ║  HOST MEM     ← CPU DRAM, transferred via PCIe (≈ 12–16 GB/s).        ║
 * ║                  h_ratings, h_sim_matrix, etc.  Transfer is expensive  ║
 * ║                  and should be minimised.                               ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WARPS  (the real hardware execution unit)                              ║
 * ║                                                                         ║
 * ║  A block's threads are grouped into WARPS of 32 threads each.         ║
 * ║  All 32 threads in a warp execute the SAME instruction at the SAME    ║
 * ║  time (SIMT — Single Instruction, Multiple Threads).                  ║
 * ║                                                                         ║
 * ║  BRANCH DIVERGENCE: if threads in the same warp take different         ║
 * ║  branches of an if-statement, the warp must execute BOTH branches      ║
 * ║  serially with some threads masked off.  This halves throughput.       ║
 * ║  In our kernels, the "if (u > v) return" causes half the warp to      ║
 * ║  idle — an accepted trade-off for code simplicity.                     ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -o cuda_rec cuda/cuda_recommender.cu -lm
 *
 *   Replace sm_75 with your GPU's compute capability:
 *     sm_60 = Pascal  (GTX 10xx)      sm_80 = Ampere (A100)
 *     sm_70 = Volta   (V100)          sm_86 = Ampere (RTX 30xx)
 *     sm_75 = Turing  (RTX 20xx)      sm_89 = Ada    (RTX 40xx)
 *
 *   To detect automatically:
 *     nvidia-smi --query-gpu=compute_cap --format=csv,noheader
 *
 * Run:
 *   ./cuda_rec              # default 1000 × 1000
 *   ./cuda_rec 2000 1500    # custom sizes
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>   /* cudaMalloc, cudaMemcpy, cudaDeviceProp, etc. */

/* ── Fixed algorithm parameters (identical across all project versions) ─── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

/* ── Runtime sizes (set from CLI in main) ───────────────────────────────── */
static int N_USERS;
static int N_ITEMS;

/*
 * HOST arrays – regular CPU heap memory (allocated with calloc).
 * Prefixed with h_ to distinguish them from the device (GPU) counterparts.
 * These are filled by the CPU and transferred to the GPU before kernel launch.
 * Results are transferred back from the GPU into these arrays after kernels.
 */
static float *h_ratings;      /* [N_USERS × N_ITEMS]  input ratings          */
static float *h_user_mean;    /* [N_USERS]             per-user mean rating   */
static float *h_sim_matrix;   /* [N_USERS × N_USERS]  Pearson similarities   */
static float *h_predictions;  /* [N_USERS × N_ITEMS]  output predictions     */

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;   /* held-out ratings for MAE evaluation          */
static int        test_size;

/* ── CUDA error-checking macro ───────────────────────────────────────────── */
/*
 * Every CUDA API function returns a cudaError_t.  Ignoring errors silently
 * leads to wrong results or crashes far from the real problem.
 * This macro wraps any CUDA call and immediately prints the file, line, and
 * human-readable error string, then exits if the call failed.
 *
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                    \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

/* ── Host-side wall-clock timer ─────────────────────────────────────────── */
/*
 * clock_gettime(CLOCK_MONOTONIC) gives nanosecond resolution on Linux.
 * We use it to time host-side work (data generation, H↔D transfers).
 * GPU kernel time is measured separately with CUDA events (see main).
 */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Host memory allocation / deallocation ───────────────────────────────── */
static void alloc_host(void)
{
    h_ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    h_user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    h_sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    h_predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

    if (!h_ratings || !h_user_mean || !h_sim_matrix || !h_predictions) {
        fprintf(stderr, "Error: host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
}

static void free_host(void)
{
    free(h_ratings); free(h_user_mean);
    free(h_sim_matrix); free(h_predictions);
    free(test_set);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 1 – Data generation  (CPU, serial)
 *
 * Identical srand(SEED) + rand() sequence to the serial/OpenMP/MPI versions.
 * Running this on the CPU avoids implementing a GPU random-number generator
 * and guarantees bit-identical datasets for cross-version MAE comparison.
 *
 * The generated h_ratings array is transferred to the GPU after this step.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void generate_data(void)
{
    srand(SEED);

    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            if ((float)rand() / RAND_MAX < SPARSITY) continue;
            float rating = (float)(rand() % 5) + 1.0f;
            if ((float)rand() / RAND_MAX < TEST_RATIO && test_size < capacity) {
                test_set[test_size].user   = u;
                test_set[test_size].item   = i;
                test_set[test_size].rating = rating;
                test_size++;
            } else {
                h_ratings[u * N_ITEMS + i] = rating;
            }
        }
    }

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *
 *  GPU KERNELS
 *
 *  A kernel is a function that runs on the GPU.  It is declared with the
 *  __global__ qualifier.  Key rules:
 *
 *   ● __global__ functions are called FROM the host but run ON the device.
 *   ● __device__ functions are called FROM the device (other kernels).
 *   ● They cannot return a value (must be void).
 *   ● They cannot use host pointers — only device pointers (d_* below).
 *   ● They cannot call most standard library functions (no printf in loops,
 *     no malloc — use device-only equivalents if needed).
 *   ● Each thread identifies itself via built-in variables:
 *       threadIdx.{x,y,z}  — thread's position within its block
 *       blockIdx.{x,y,z}   — block's position within the grid
 *       blockDim.{x,y,z}   — number of threads per block in each dimension
 *       gridDim.{x,y,z}    — number of blocks in the grid in each dimension
 *
 * ═══════════════════════════════════════════════════════════════════════════ */


/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 1 – kernel_user_means
 *
 * PURPOSE: compute the mean rating for each user.
 *
 * PARALLELISM: 1-D grid.  One thread per user.
 *   Thread 0 processes user 0, thread 1 processes user 1, etc.
 *
 * LAUNCH: <<<ceil(N_USERS/256), 256>>>
 *   If N_USERS = 1000 and block size = 256, we launch 4 blocks × 256 = 1024
 *   threads.  Threads 1000–1023 have u >= N_USERS and return immediately.
 *
 * GLOBAL MEMORY ACCESS PATTERN:
 *   Thread u reads ratings[u*N_ITEMS + 0], ratings[u*N_ITEMS + 1], …
 *   Adjacent threads u and u+1 read ratings[u*N_ITEMS] and
 *   ratings[(u+1)*N_ITEMS] which are N_ITEMS floats apart — NOT coalesced.
 *   For this kernel the access pattern is row-major (each thread scans its
 *   own row), which is not ideal for coalescing but unavoidable given the
 *   data layout.  The kernel is fast enough because it is memory-bound only
 *   at the O(N_USERS × N_ITEMS) scale, a much smaller cost than similarity.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_user_means(const float *ratings, float *user_mean,
                                   int N_USERS, int N_ITEMS)
{
    /*
     * Compute this thread's global user index.
     *
     * Formula:  global_id = blockIdx.x * blockDim.x + threadIdx.x
     *
     * Example with blockDim.x=256:
     *   Block 0: threads 0..255   → u = 0..255
     *   Block 1: threads 0..255   → u = 256..511
     *   Block 2: threads 0..255   → u = 512..767
     *   Block 3: threads 0..255   → u = 768..1023  (1000-1023 are guards)
     */
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    /* Guard: threads beyond the last user do nothing. */
    if (u >= N_USERS) return;

    /*
     * Each thread accumulates its own private sum and count in registers.
     * No shared memory or atomic operations needed because no two threads
     * write to the same index of user_mean[].
     */
    double sum = 0.0;
    int    cnt = 0;
    for (int i = 0; i < N_ITEMS; i++) {
        float r = ratings[u * N_ITEMS + i];
        if (r != 0.0f) { sum += r; cnt++; }
    }

    /* Write result — each thread writes to its own unique index. */
    user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2 – kernel_similarity
 *
 * PURPOSE: compute the Pearson correlation between every pair of users (u,v).
 *
 * PARALLELISM: 2-D grid.  One thread per (u, v) pair.
 *   The similarity matrix is N_USERS × N_USERS.  We map:
 *     threadIdx.x / blockIdx.x → user index u   (row)
 *     threadIdx.y / blockIdx.y → user index v   (column)
 *
 * WHY 2-D GRID?
 *   A 1-D grid could also work, but a 2-D grid maps cleanly onto the 2-D
 *   problem.  Each thread directly computes sim_matrix[u][v] without needing
 *   to unpack a linear index into a row and column.
 *
 * UPPER-TRIANGLE ONLY:
 *   Since sim(u,v) == sim(v,u), we only compute pairs with u < v.
 *   The thread for (u,v) with u < v writes BOTH sim_matrix[u][v] and
 *   sim_matrix[v][u] in one go.  Threads with u > v return immediately.
 *   Threads with u == v set the diagonal to 1.0.
 *
 *   This halves the total arithmetic but introduces warp divergence in blocks
 *   that straddle the diagonal.  For large N_USERS the savings dominate.
 *
 * LAUNCH: <<<dim3(ceil(N/16), ceil(N/16)), dim3(16,16)>>>
 *   16×16 = 256 threads per block.  256 is a common sweet spot: enough
 *   threads to hide global memory latency, not so many that register usage
 *   reduces SM occupancy.
 *
 * RACE-FREE WRITES:
 *   Thread (u,v) with u < v writes to sim_matrix[u*N_USERS+v] and
 *   sim_matrix[v*N_USERS+u].  No other thread writes to those two cells
 *   (the (v,u) thread returns early because v > u).  So there is no race.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_similarity(const float *ratings, const float *user_mean,
                                   float *sim_matrix, int N_USERS, int N_ITEMS)
{
    /*
     * 2-D thread index calculation.
     *
     * blockIdx.x and threadIdx.x together identify the column (u) dimension.
     * blockIdx.y and threadIdx.y together identify the row    (v) dimension.
     *
     * With blockDim = (16,16) and a 1000×1000 matrix we launch a 63×63 grid
     * of blocks (63 × 16 = 1008 ≥ 1000).  The excess threads (u or v ≥ 1000)
     * are caught by the guard below.
     */
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    /* Guard: discard out-of-bounds threads. */
    if (u >= N_USERS || v >= N_USERS) return;

    /* Self-similarity is always 1 by definition. */
    if (u == v) { sim_matrix[u * N_USERS + v] = 1.0f; return; }

    /*
     * Skip the lower triangle.  The thread that handles (v, u) where v < u
     * will compute the value and write BOTH cells.  This thread does nothing.
     */
    if (u > v) return;

    /* ── Pearson correlation for the co-rated items of users u and v ── */
    float  mu = user_mean[u], mv = user_mean[v];
    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int    co  = 0;   /* count of items rated by BOTH u and v */

    for (int i = 0; i < N_ITEMS; i++) {
        float ru = ratings[u * N_ITEMS + i];
        float rv = ratings[v * N_ITEMS + i];

        /*
         * Both ratings must be non-zero — zero encodes "not rated", not a
         * rating of 0.  Only co-rated items contribute to the correlation.
         */
        if (ru != 0.0f && rv != 0.0f) {
            double du = (double)ru - mu;   /* deviation from user u's mean */
            double dv = (double)rv - mv;   /* deviation from user v's mean */
            num   += du * dv;              /* numerator:   Σ(du × dv)      */
            den_u += du * du;              /* denominator: Σ(du²)           */
            den_v += dv * dv;              /* denominator: Σ(dv²)           */
            co++;
        }
    }

    /*
     * Pearson formula:  r = Σ(du·dv) / (√Σdu² · √Σdv²)
     *
     * At least 2 co-rated items are required for a meaningful correlation.
     * If denom ≈ 0, one user rated all items the same (zero variance) →
     * correlation is undefined; treat it as 0 (no similarity signal).
     */
    float s = 0.0f;
    if (co >= 2) {
        double denom = sqrt(den_u) * sqrt(den_v);
        if (denom >= 1e-10) {
            s = (float)(num / denom);
            /* Clamp to [-1, 1] to correct floating-point rounding errors. */
            if (s >  1.0f) s =  1.0f;
            if (s < -1.0f) s = -1.0f;
        }
    }

    /*
     * Write both symmetric cells in a single thread — avoids a second kernel
     * pass to mirror the lower triangle.  Both writes go to different cache
     * lines, which is fine since no other thread touches these cells.
     */
    sim_matrix[u * N_USERS + v] = s;
    sim_matrix[v * N_USERS + u] = s;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 3 – kernel_predictions
 *
 * PURPOSE: predict the rating for every (user, item) pair using collaborative
 *          filtering with the TOP_K most similar neighbours.
 *
 * PARALLELISM: 2-D grid.  One thread per (user, item) pair.
 *   With N_USERS = N_ITEMS = 1000, this launches 1,000,000 threads.
 *
 * LAUNCH: <<<dim3(ceil(N_USERS/16), ceil(N_ITEMS/16)), dim3(16,16)>>>
 *
 * TOP-K SELECTION — in-register min-heap approach:
 *   The CPU version uses qsort(), which is not available in device code.
 *   Instead, each thread maintains three fixed-size arrays in its REGISTERS:
 *     top_sim[TOP_K]  — similarity of the k-th best neighbour
 *     top_rat[TOP_K]  — that neighbour's rating for this item
 *     top_mu[TOP_K]   — that neighbour's mean rating
 *   plus a running pointer to the WORST (smallest) entry in the buffer.
 *
 *   Algorithm:
 *     1. Fill the buffer with the first TOP_K eligible neighbours.
 *     2. For each subsequent neighbour v:
 *        - if sim(u,v) > worst entry in the buffer, replace the worst entry.
 *        - Rescan the buffer to find the new worst entry.
 *   This is O(N_USERS × TOP_K) per thread — acceptable because TOP_K = 20.
 *
 * REGISTER PRESSURE:
 *   3 × TOP_K = 60 floats + scalars ≈ 64 registers per thread.
 *   With 256 threads/block (16×16), the SM needs 256 × 64 = 16,384 registers.
 *   Modern GPUs have 65,536 registers per SM — this fits comfortably and
 *   allows 4 concurrent blocks per SM (good occupancy).
 *   If TOP_K were much larger, arrays would spill to local memory (DRAM),
 *   degrading performance.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_predictions(const float *ratings, const float *user_mean,
                                    const float *sim_matrix, float *predictions,
                                    int N_USERS, int N_ITEMS)
{
    /* Map (blockIdx, threadIdx) to a unique (user, item) pair. */
    int u    = blockIdx.x * blockDim.x + threadIdx.x;
    int item = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= N_USERS || item >= N_ITEMS) return;

    /*
     * If the user already rated this item, their actual rating is the best
     * "prediction" — no collaborative filtering needed.
     */
    float r_ui = ratings[u * N_ITEMS + item];
    if (r_ui != 0.0f) {
        predictions[u * N_ITEMS + item] = r_ui;
        return;
    }

    /* ── Top-K neighbour selection using a fixed in-register buffer ── */

    /*
     * These arrays hold the current best TOP_K neighbours found so far.
     * Because TOP_K is a compile-time constant (#define), the compiler
     * can allocate these as register arrays rather than local memory.
     */
    float top_sim[TOP_K];
    float top_rat[TOP_K];
    float top_mu [TOP_K];
    int   top_cnt = 0;     /* how many slots are currently filled            */
    float worst   = -2.0f; /* sim value of the weakest neighbour in buffer  */
    int   worst_j = 0;     /* index of the weakest neighbour in the buffer  */

    for (int v = 0; v < N_USERS; v++) {
        if (v == u) continue;

        /* Skip neighbours who haven't rated this item. */
        float rv = ratings[v * N_ITEMS + item];
        if (rv == 0.0f) continue;

        /* Skip neighbours with zero or negative similarity. */
        float s = sim_matrix[u * N_USERS + v];
        if (s <= 0.0f) continue;

        if (top_cnt < TOP_K) {
            /*
             * Buffer not yet full — just insert and track the new worst.
             * On the very first insertion (top_cnt == 0) any s qualifies
             * as the worst; afterwards update only if s is smaller.
             */
            top_sim[top_cnt] = s;
            top_rat[top_cnt] = rv;
            top_mu [top_cnt] = user_mean[v];
            if (top_cnt == 0 || s < worst) { worst = s; worst_j = top_cnt; }
            top_cnt++;
        } else if (s > worst) {
            /*
             * Buffer is full and this neighbour is better than the weakest
             * one in the buffer.  Evict the weakest and insert the new one.
             */
            top_sim[worst_j] = s;
            top_rat[worst_j] = rv;
            top_mu [worst_j] = user_mean[v];

            /* Rescan the buffer to find the new worst slot. */
            worst = top_sim[0]; worst_j = 0;
            for (int j = 1; j < TOP_K; j++) {
                if (top_sim[j] < worst) { worst = top_sim[j]; worst_j = j; }
            }
        }
    }

    /* Fall back to the user's mean if no neighbours were found. */
    float mu_u = user_mean[u];
    if (top_cnt == 0) { predictions[u * N_ITEMS + item] = mu_u; return; }

    /*
     * Weighted-average prediction formula:
     *
     *   pred(u, item) = mean(u) + Σ[ sim(u,v) × (rating(v,item) - mean(v)) ]
     *                             ─────────────────────────────────────────
     *                                          Σ sim(u,v)
     *
     * The deviation term (rating - mean) adjusts for the fact that some users
     * rate everything high or low; it measures how much user v's rating of
     * this item DEVIATES from their typical rating behaviour.
     */
    double num = 0.0, den = 0.0;
    for (int j = 0; j < top_cnt; j++) {
        double s = top_sim[j];
        num += s * ((double)top_rat[j] - top_mu[j]);
        den += s;
    }

    float pred = (den > 1e-10) ? mu_u + (float)(num / den) : mu_u;

    /* Clamp to the valid rating scale [1, 5]. */
    if (pred < 1.0f) pred = 1.0f;
    if (pred > 5.0f) pred = 5.0f;
    predictions[u * N_ITEMS + item] = pred;
}


/* ── Host-side evaluation helpers ────────────────────────────────────────── */

/*
 * evaluate_mae() runs on the CPU after h_predictions is copied back from the
 * GPU.  It iterates over the held-out test ratings and computes:
 *   MAE = (1/|T|) × Σ |predicted - actual|
 */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user, i = test_set[t].item;
        err += fabs(h_predictions[u * N_ITEMS + i] - test_set[t].rating);
    }
    return (float)(err / test_size);
}

/*
 * similarity_checksum() sums every element of h_sim_matrix.
 * This value must match the serial / OpenMP / Pthreads / MPI checksums
 * for the same N_USERS / N_ITEMS / SEED, confirming numerical correctness.
 */
static double similarity_checksum(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += h_sim_matrix[u * N_USERS + v];
    return s;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 *
 * The host program that orchestrates:
 *   1. CPU data generation
 *   2. Device memory allocation   (cudaMalloc)
 *   3. Host → Device transfer     (cudaMemcpy H→D)
 *   4. GPU kernel launches        (<<<grid, block>>>)
 *   5. Device → Host transfer     (cudaMemcpy D→H)
 *   6. CPU evaluation and output
 *   7. Device memory cleanup      (cudaFree)
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * cudaGetDeviceProperties() queries the GPU's hardware specs.
     * This is optional but useful for verifying the correct GPU is active
     * and for understanding the hardware context of timing results.
     *
     * prop.major / prop.minor = compute capability (e.g. 7.5 for Turing).
     * prop.multiProcessorCount = number of Streaming Multiprocessors (SMs).
     *   More SMs = more blocks can run simultaneously.
     */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));   /* device 0 = first GPU */

    printf("=== Pearson Correlation Recommender – CUDA Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d\n", N_USERS, N_ITEMS, TOP_K);
    printf("    GPU: %s  |  SMs: %d  |  Compute: %d.%d\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    alloc_host();

    double t0, t1;

    /* ── Phase 1: Data generation on the CPU ────────────────────────────── */
    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* ── Device (GPU) memory allocation ─────────────────────────────────── */
    /*
     * cudaMalloc allocates memory in DEVICE DRAM (the GPU's own RAM).
     * The resulting pointers (d_*) are NOT usable on the CPU — dereferencing
     * them on the host causes a segfault or undefined behaviour.
     *
     * We allocate:
     *   d_ratings   — the input matrix copied from h_ratings
     *   d_user_mean — filled by kernel_user_means
     *   d_sim       — filled by kernel_similarity
     *   d_pred      — filled by kernel_predictions, then copied back
     */
    float *d_ratings, *d_user_mean, *d_sim, *d_pred;
    CUDA_CHECK(cudaMalloc(&d_ratings,  (size_t)N_USERS * N_ITEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_user_mean, (size_t)N_USERS           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sim,       (size_t)N_USERS * N_USERS  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred,      (size_t)N_USERS * N_ITEMS  * sizeof(float)));

    /*
     * cudaMemset fills device memory with a byte value — like C memset but
     * on the GPU.  We zero d_sim and d_pred so that any cell not written by
     * a kernel (e.g. guard threads that returned early) is safely 0.
     */
    CUDA_CHECK(cudaMemset(d_sim,  0, (size_t)N_USERS * N_USERS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pred, 0, (size_t)N_USERS * N_ITEMS * sizeof(float)));

    /* ── Host → Device transfer: ratings matrix ─────────────────────────── */
    /*
     * cudaMemcpy copies data between host and device memory.
     * The fourth argument is the direction flag:
     *   cudaMemcpyHostToDevice  — CPU RAM → GPU DRAM
     *   cudaMemcpyDeviceToHost  — GPU DRAM → CPU RAM
     *   cudaMemcpyDeviceToDevice — GPU DRAM → GPU DRAM (stays on GPU)
     *
     * PCIe bandwidth is ~12-16 GB/s — transfers are expensive.
     * Minimise them by keeping data on the GPU across multiple kernels,
     * which we do here (ratings stays on device for all three kernels).
     */
    t0 = now_sec();
    CUDA_CHECK(cudaMemcpy(d_ratings, h_ratings,
                          (size_t)N_USERS * N_ITEMS * sizeof(float),
                          cudaMemcpyHostToDevice));
    t1 = now_sec();
    printf("[Timing] H→D ratings        : %.4f s\n", t1 - t0);

    /* ── CUDA event timers ───────────────────────────────────────────────── */
    /*
     * CUDA Events are the correct way to time GPU kernels.
     *
     * Why not clock_gettime()?
     *   Kernel launches are ASYNCHRONOUS — the CPU call returns immediately
     *   while the GPU is still running.  clock_gettime() on the host after
     *   a kernel launch measures near-zero time because the GPU work hasn't
     *   finished yet.
     *
     * How events work:
     *   cudaEventRecord(ev, stream) inserts a timestamp marker into the GPU's
     *   command queue.  The GPU records the actual time when it processes that
     *   marker in execution order.
     *   cudaEventSynchronize(ev_stop) BLOCKS the CPU until the GPU has
     *   processed ev_stop (and therefore finished the kernel).
     *   cudaEventElapsedTime(&ms, ev_start, ev_stop) gives the kernel's
     *   wall-clock duration in milliseconds.
     */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    float ms;   /* elapsed time in milliseconds, filled by cudaEventElapsedTime */

    /* ── Phase 2: User means ─────────────────────────────────────────────── */
    {
        /*
         * 1-D launch configuration.
         * block = 256 threads per block (common choice; must be multiple of 32).
         * grid  = ceiling(N_USERS / 256) blocks to cover all users.
         */
        int block = 256;
        int grid  = (N_USERS + block - 1) / block;

        CUDA_CHECK(cudaEventRecord(ev_start));

        /*
         * Kernel launch syntax:  functionName<<<grid, block>>>(args...)
         *
         *   grid  — dim3 or int: number of blocks in x (and optionally y, z)
         *   block — dim3 or int: number of threads per block in each dimension
         *
         * This call is ASYNCHRONOUS on the CPU side — it returns immediately
         * and the GPU executes the kernel in the background.
         */
        kernel_user_means<<<grid, block>>>(d_ratings, d_user_mean,
                                           N_USERS, N_ITEMS);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));   /* wait for GPU to finish */

        /*
         * cudaGetLastError() checks whether the kernel launch itself failed
         * (e.g. invalid launch configuration, out-of-resources).  This must
         * be checked AFTER synchronisation, because kernel errors are reported
         * asynchronously.
         */
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        printf("[Timing] User mean compute  : %.4f s  [GPU kernel]\n",
               ms / 1000.0f);

        /* Copy means back to the host for the host-side MAE computation. */
        CUDA_CHECK(cudaMemcpy(h_user_mean, d_user_mean,
                              N_USERS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        /* d_user_mean also stays on the device for use by kernel_similarity. */
    }

    /* ── Phase 3: Similarity matrix ──────────────────────────────────────── */
    double t_sim;
    {
        /*
         * 2-D launch configuration for the N_USERS × N_USERS similarity matrix.
         *
         * block = dim3(16, 16) = 256 threads per block arranged in a 16×16 tile.
         * grid  = dim3(ceil(N/16), ceil(N/16)) tiles to cover the full matrix.
         *
         * For N_USERS = 1000:
         *   grid = dim3(63, 63) = 3969 blocks × 256 threads = 1,016,064 threads.
         *   (Extra threads with u≥1000 or v≥1000 are eliminated by the guard.)
         */
        dim3 block(16, 16);
        dim3 grid((N_USERS + 15) / 16, (N_USERS + 15) / 16);

        CUDA_CHECK(cudaEventRecord(ev_start));
        kernel_similarity<<<grid, block>>>(d_ratings, d_user_mean,
                                           d_sim, N_USERS, N_ITEMS);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        t_sim = ms / 1000.0;
        printf("[Timing] Similarity matrix  : %.4f s  [GPU kernel]\n", t_sim);

        /* Copy full sim_matrix back to the host for the checksum print. */
        CUDA_CHECK(cudaMemcpy(h_sim_matrix, d_sim,
                              (size_t)N_USERS * N_USERS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());
        /* d_sim stays on the device for use by kernel_predictions. */
    }

    /* ── Phase 4: Predictions ─────────────────────────────────────────────── */
    double t_pred;
    {
        /*
         * 2-D launch: one thread per (user, item) pair.
         * grid covers N_USERS × N_ITEMS.
         *
         * For N_USERS=N_ITEMS=1000: 63×63 blocks × 256 threads = ~1M threads.
         * Each thread independently predicts one cell of the output matrix.
         */
        dim3 block(16, 16);
        dim3 grid((N_USERS + 15) / 16, (N_ITEMS + 15) / 16);

        CUDA_CHECK(cudaEventRecord(ev_start));
        kernel_predictions<<<grid, block>>>(d_ratings, d_user_mean,
                                            d_sim, d_pred, N_USERS, N_ITEMS);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        t_pred = ms / 1000.0;
        printf("[Timing] Prediction phase   : %.4f s  [GPU kernel]\n", t_pred);

        /* Transfer predictions from GPU to CPU for evaluation and display. */
        CUDA_CHECK(cudaMemcpy(h_predictions, d_pred,
                              (size_t)N_USERS * N_ITEMS * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    /* ── Phase 5: Evaluation (CPU, serial) ───────────────────────────────── */
    printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
           evaluate_mae(), test_size);
    printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);

    /* Sample output */
    int show_u = (N_USERS < 5) ? N_USERS : 5;
    int show_i = (N_ITEMS < 5) ? N_ITEMS : 5;
    printf("\n--- Sample Predictions (first %d users, %d items) ---\n",
           show_u, show_i);
    printf("%-9s", "User\\Item");
    for (int i = 0; i < show_i; i++) printf("  Item%-3d", i);
    printf("\n");
    for (int u = 0; u < show_u; u++) {
        printf("User %-4d", u);
        for (int i = 0; i < show_i; i++)
            printf("  %5.2f  ", h_predictions[u * N_ITEMS + i]);
        printf("\n");
    }

    /* ── Device memory cleanup ────────────────────────────────────────────── */
    /*
     * cudaFree releases device memory.  Always free device allocations before
     * the program exits to prevent resource leaks (especially important in
     * long-running servers or when multiple CUDA contexts share a GPU).
     * cudaEventDestroy releases the event objects similarly.
     */
    CUDA_CHECK(cudaFree(d_ratings));
    CUDA_CHECK(cudaFree(d_user_mean));
    CUDA_CHECK(cudaFree(d_sim));
    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free_host();
    return 0;
}
