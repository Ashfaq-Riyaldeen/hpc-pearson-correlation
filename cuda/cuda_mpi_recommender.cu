/* Hybrid CUDA+MPI Distributed GPU Pearson Correlation Recommender
 * 
 * Each MPI process controls its own GPU
 * 
 * Compile:
 *   mpicxx -O2 -o cuda_mpi_rec cuda_mpi_recommender.cu -lm -L/usr/local/cuda/lib64 -lcudart
 *   OR: mpicc -O2 -o cuda_mpi_rec cuda_mpi_recommender.cu -lm -L/usr/local/cuda/lib64 -lcudart
 * 
 * Run:
 *   mpirun -np 2 ./cuda_mpi_rec
 *   mpirun -np 4 ./cuda_mpi_rec 500 300
 *   mpirun -np 4 ./cuda_mpi_rec 2000 1500
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ── Fixed parameters ────────────────────────────────────────────────────── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f
#define BLOCK_SIZE      256

/* ── MPI variables ───────────────────────────────────────────────────────── */
static int rank, num_procs;
static MPI_Comm comm = MPI_COMM_WORLD;

/* ── Runtime size variables (set in main) ────────────────────────────────── */
static int N_USERS;
static int N_ITEMS;

/* ── Host arrays ─────────────────────────────────────────────────────────── */
static float *ratings_h;           /* [N_USERS × N_ITEMS] – rank 0 only      */
static float *user_mean_h;         /* [N_USERS] – all ranks                   */
static float *sim_matrix_h;        /* [N_USERS × N_USERS] – all ranks        */
static float *predictions_h;       /* [N_USERS × N_ITEMS] – all ranks        */
static float *local_sim_rows_h;    /* [local_num_users × N_USERS] – this rank*/

/* ── Device arrays ───────────────────────────────────────────────────────── */
static float *ratings_d;           /* [N_USERS × N_ITEMS]                    */
static float *user_mean_d;         /* [N_USERS]                              */
static float *sim_matrix_d;        /* [N_USERS × N_USERS]                    */
static float *predictions_d;       /* [N_USERS × N_ITEMS]                    */
static float *local_sim_rows_d;    /* [local_num_users × N_USERS]            */

/* ── Local computation ranges ────────────────────────────────────────────── */
static int local_user_start;
static int local_user_end;
static int local_num_users;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set_h;
static int        test_size;

/* ── Convenience macros (host) ───────────────────────────────────────────── */
#define R_H(u,i)    ratings_h[(u)*N_ITEMS + (i)]
#define SIM_H(u,v)  sim_matrix_h[(u)*N_USERS + (v)]
#define PRED_H(u,i) predictions_h[(u)*N_ITEMS + (i)]

/* ── CUDA utilities ──────────────────────────────────────────────────────── */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Rank %d] CUDA error: %s\n", rank, \
                    cudaGetErrorString(err)); \
            MPI_Abort(comm, EXIT_FAILURE); \
        } \
    } while(0)

/* ── Timing ──────────────────────────────────────────────────────────────── */
static inline double now_sec(void)
{
    return MPI_Wtime();
}

/* ── Initialize MPI GPU binding ──────────────────────────––––––––––––––––– */
static void init_gpu(void)
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "[Rank %d] No CUDA devices found!\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    if (rank == 0) {
        printf("[GPU] %d devices found. Rank %d -> GPU %d\n",
               device_count, rank, device_id);
    }
}

/* ── Allocate memory ─────────────────────────────────────────────––––––––– */
static void alloc_arrays(void)
{
    if (rank == 0) {
        ratings_h = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
        if (!ratings_h) {
            fprintf(stderr, "[Rank 0] Host ratings allocation failed.\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    user_mean_h   = (float *)calloc(N_USERS,            sizeof(float));
    sim_matrix_h  = (float *)calloc(N_USERS * N_USERS,  sizeof(float));
    predictions_h = (float *)calloc(N_USERS * N_ITEMS,  sizeof(float));
    local_sim_rows_h = (float *)calloc(local_num_users * N_USERS, sizeof(float));

    if (!user_mean_h || !sim_matrix_h || !predictions_h || !local_sim_rows_h) {
        fprintf(stderr, "[Rank %d] Host memory allocation failed.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Device memory */
    int size_ratings = N_USERS * N_ITEMS * sizeof(float);
    int size_users   = N_USERS * sizeof(float);
    int size_sim     = N_USERS * N_USERS * sizeof(float);
    int size_local_sim = local_num_users * N_USERS * sizeof(float);

    CUDA_CHECK(cudaMalloc(&ratings_d,     size_ratings));
    CUDA_CHECK(cudaMalloc(&user_mean_d,   size_users));
    CUDA_CHECK(cudaMalloc(&sim_matrix_d,  size_sim));
    CUDA_CHECK(cudaMalloc(&predictions_d, size_ratings));
    CUDA_CHECK(cudaMalloc(&local_sim_rows_d, size_local_sim));
}

static void free_arrays(void)
{
    if (rank == 0) free(ratings_h);
    free(user_mean_h);
    free(sim_matrix_h);
    free(predictions_h);
    free(local_sim_rows_h);
    free(test_set_h);

    cudaFree(ratings_d);
    cudaFree(user_mean_d);
    cudaFree(sim_matrix_d);
    cudaFree(predictions_d);
    cudaFree(local_sim_rows_d);
}

/* ── Data generation (rank 0 only) ───────────────–––––––––––––––––––––––– */
static void generate_data(void)
{
    if (rank != 0) return;

    srand(SEED);

    int capacity = (int)(N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set_h = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            if ((float)rand() / RAND_MAX < SPARSITY) continue;

            float rating = (float)(rand() % 5) + 1.0f;

            if ((float)rand() / RAND_MAX < TEST_RATIO && test_size < capacity) {
                test_set_h[test_size].user   = u;
                test_set_h[test_size].item   = i;
                test_set_h[test_size].rating = rating;
                test_size++;
            } else {
                R_H(u, i) = rating;
            }
        }
    }

    printf("[Data] Users: %d | Items: %d | Sparsity: %.0f%% | Test: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ── Broadcast data ──────────────────────────────–––––––––––––––––––––––– */
static void broadcast_data(void)
{
    int size_ratings = N_USERS * N_ITEMS * sizeof(float);
    if (rank == 0) {
        MPI_Bcast(ratings_h, size_ratings, MPI_BYTE, 0, comm);
    } else {
        ratings_h = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
        MPI_Bcast(ratings_h, size_ratings, MPI_BYTE, 0, comm);
    }
}

/* ── Compute user means (host) ───────────────–––––––––––––––––––––––––– */
static void compute_user_means_host(void)
{
    for (int u = 0; u < N_USERS; u++) {
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R_H(u, i) != 0.0f) { sum += R_H(u, i); cnt++; }
        }
        user_mean_h[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }

    MPI_Bcast(user_mean_h, N_USERS, MPI_FLOAT, 0, comm);
}

/* ── CUDA Kernel: Local Pearson similarities ──–––––––––––––––––––––––––– */
__global__ void kernel_local_pearson(
    const float *ratings, const float *user_means,
    float *local_sim_rows, int local_start, int N_USERS, int N_ITEMS,
    int local_num_users)
{
    int local_u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_u >= local_num_users || v >= N_USERS) return;

    int u = local_start + local_u;

    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int co = 0;
    float mu = user_means[u];
    float mv = user_means[v];

    for (int i = 0; i < N_ITEMS; i++) {
        float ru = ratings[u * N_ITEMS + i];
        float rv = ratings[v * N_ITEMS + i];

        if (ru != 0.0f && rv != 0.0f) {
            double du = ru - mu;
            double dv = rv - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    if (co < 2) {
        local_sim_rows[local_u * N_USERS + v] = 0.0f;
        return;
    }

    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) {
        local_sim_rows[local_u * N_USERS + v] = 0.0f;
        return;
    }

    float s = (float)(num / denom);
    s = fmax(-1.0f, fmin(1.0f, s));
    local_sim_rows[local_u * N_USERS + v] = s;
}

/* ── CUDA Kernel: Local predictions ──––––––––––––––––––––––––––––––––––– */
__global__ void kernel_local_predictions(
    const float *ratings, const float *user_means, const float *sim_matrix,
    float *predictions, int local_start, int N_USERS, int N_ITEMS, int TOP_K,
    int local_num_users)
{
    int local_u = blockIdx.x * blockDim.x + threadIdx.x;
    int item = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_u >= local_num_users || item >= N_ITEMS) return;

    int u = local_start + local_u;
    float rating = ratings[u * N_ITEMS + item];

    if (rating != 0.0f) {
        predictions[u * N_ITEMS + item] = rating;
        return;
    }

    float best_sims[32];
    int best_idxs[32];
    int k = 0;

    for (int v = 0; v < N_USERS; v++) {
        if (v == u || ratings[v * N_ITEMS + item] == 0.0f) continue;

        float s = sim_matrix[u * N_USERS + v];
        if (s <= 0.0f) continue;

        if (k < TOP_K) {
            best_sims[k] = s;
            best_idxs[k] = v;
            k++;
        } else {
            int min_idx = 0;
            for (int j = 1; j < TOP_K; j++) {
                if (best_sims[j] < best_sims[min_idx]) min_idx = j;
            }
            if (s > best_sims[min_idx]) {
                best_sims[min_idx] = s;
                best_idxs[min_idx] = v;
            }
        }
    }

    if (k == 0) {
        predictions[u * N_ITEMS + item] = user_means[u];
        return;
    }

    double num = 0.0, den = 0.0;
    for (int j = 0; j < k; j++) {
        float s = best_sims[j];
        int v = best_idxs[j];
        num += s * (ratings[v * N_ITEMS + item] - user_means[v]);
        den += s;
    }

    float pred = (den > 1e-10)
                 ? user_means[u] + (float)(num / den)
                 : user_means[u];
    pred = fmax(1.0f, fmin(5.0f, pred));
    predictions[u * N_ITEMS + item] = pred;
}

/* ── Host: Evaluate MAE ––––––––––––––––––––––––––––––––––––––––––––––––– */
static float evaluate_mae_host(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED_H(test_set_h[t].user, test_set_h[t].item)
                    - test_set_h[t].rating);
    
    double global_err;
    MPI_Reduce(&err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    
    if (rank == 0 && test_size > 0)
        return (float)(global_err / test_size);
    return 0.0f;
}

static double similarity_checksum_host(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM_H(u, v);
    return s;
}

/* ── Main ────────────────────────────────────––––––––––––––––––––––––––– */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        if (rank == 0) fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Compute local user ranges */
    int base_users = N_USERS / num_procs;
    int extra_users = N_USERS % num_procs;
    local_num_users = base_users + (rank < extra_users ? 1 : 0);
    local_user_start = rank * base_users + (rank < extra_users ? rank : extra_users);
    local_user_end = local_user_start + local_num_users;

    if (rank == 0) {
        printf("=== Pearson Correlation Recommender – Hybrid CUDA+MPI ===\n");
        printf("    Processes: %d | Users: %d | Items: %d | Top-K: %d\n\n",
               num_procs, N_USERS, N_ITEMS, TOP_K);
    }

    init_gpu();
    alloc_arrays();

    double t0, t1, t_sim, t_pred;

    t0 = now_sec(); generate_data(); MPI_Barrier(comm); t1 = now_sec();
    if (rank == 0) printf("[Timing] Data generation       : %.4f s\n", t1 - t0);

    t0 = now_sec(); broadcast_data(); t1 = now_sec();
    if (rank == 0) printf("[Timing] Data broadcast        : %.4f s\n", t1 - t0);

    t0 = now_sec(); compute_user_means_host(); t1 = now_sec();
    if (rank == 0) printf("[Timing] User means compute    : %.4f s\n", t1 - t0);

    /* Copy to device */
    t0 = now_sec();
    int size_ratings = N_USERS * N_ITEMS * sizeof(float);
    int size_users   = N_USERS * sizeof(float);
    int size_sim     = N_USERS * N_USERS * sizeof(float);
    int size_local_sim = local_num_users * N_USERS * sizeof(float);

    CUDA_CHECK(cudaMemcpy(ratings_d,   ratings_h,   size_ratings, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(user_mean_d, user_mean_h, size_users,   cudaMemcpyHostToDevice));
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Host-to-device copy   : %.4f s\n", t1 - t0);

    /* Compute local similarities on GPU */
    t0 = now_sec();
    int grid_dim = (local_num_users + 16 - 1) / 16;
    int grid_dim2 = (N_USERS + 16 - 1) / 16;
    dim3 grid(grid_dim, grid_dim2);
    dim3 block(16, 16);
    kernel_local_pearson<<<grid, block>>>(
        ratings_d, user_mean_d, local_sim_rows_d, local_user_start,
        N_USERS, N_ITEMS, local_num_users);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = now_sec();
    t_sim = t1 - t0;
    if (rank == 0) printf("[Timing] Local similarities GPU : %.4f s\n", t_sim);

    /* Allgatherv similarities */
    t0 = now_sec();
    int *sendcounts = (int *)malloc(num_procs * sizeof(int));
    int *displs = (int *)malloc(num_procs * sizeof(int));

    for (int i = 0; i < num_procs; i++) {
        int n = N_USERS / num_procs + (i < (N_USERS % num_procs) ? 1 : 0);
        sendcounts[i] = n * N_USERS;
        displs[i] = (i == 0) ? 0 : (displs[i-1] + sendcounts[i-1]);
    }

    CUDA_CHECK(cudaMemcpy(local_sim_rows_h, local_sim_rows_d, size_local_sim, cudaMemcpyDeviceToHost));
    MPI_Allgatherv(local_sim_rows_h, sendcounts[rank], MPI_FLOAT,
                   sim_matrix_h, sendcounts, displs, MPI_FLOAT, comm);
    CUDA_CHECK(cudaMemcpy(sim_matrix_d, sim_matrix_h, size_sim, cudaMemcpyHostToDevice));
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Similarity gather     : %.4f s\n", t1 - t0);

    /* Compute predictions on GPU */
    t0 = now_sec();
    grid_dim = (local_num_users + 16 - 1) / 16;
    grid_dim2 = (N_ITEMS + 16 - 1) / 16;
    dim3 grid_pred(grid_dim, grid_dim2);
    kernel_local_predictions<<<grid_pred, block>>>(
        ratings_d, user_mean_d, sim_matrix_d, predictions_d, local_user_start,
        N_USERS, N_ITEMS, TOP_K, local_num_users);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = now_sec();
    t_pred = t1 - t0;
    if (rank == 0) printf("[Timing] Local predictions GPU : %.4f s\n", t_pred);

    /* Gather predictions */
    t0 = now_sec();
    CUDA_CHECK(cudaMemcpy(predictions_h, predictions_d, size_ratings, cudaMemcpyDeviceToHost));
    int pred_sendcount = local_num_users * N_ITEMS;
    MPI_Allgatherv(predictions_h + local_user_start * N_ITEMS, pred_sendcount, MPI_FLOAT,
                   predictions_h, sendcounts, displs, MPI_FLOAT, comm);
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Prediction gather     : %.4f s\n", t1 - t0);

    /* Evaluate */
    float mae = evaluate_mae_host();
    double checksum = similarity_checksum_host();

    if (rank == 0) {
        printf("\n[Results]\n");
        printf("  MAE on test set              : %.6f\n", mae);
        printf("  Sim-matrix checksum          : %.6f\n", checksum);
        printf("  Total (sim+pred)             : %.4f s\n", t_sim + t_pred);
    }

    free(sendcounts);
    free(displs);
    free_arrays();

    MPI_Finalize();
    return EXIT_SUCCESS;
}
