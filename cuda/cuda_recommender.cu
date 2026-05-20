/* CUDA GPU-Accelerated Pearson Correlation Recommender
 * 
 * Compile:
 *   nvcc -O2 -o cuda_rec cuda_recommender.cu -lm
 * 
 * Run:
 *   ./cuda_rec
 *   ./cuda_rec 500 300
 *   ./cuda_rec 2000 1500
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Fixed parameters ────────────────────────────────────────────────────── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f
#define BLOCK_SIZE      256

/* ── Runtime size variables (set in main) ────────────────────────────────── */
static int N_USERS;
static int N_ITEMS;

/* ── Host arrays ─────────────────────────────────────────────────────────── */
static float *ratings_h;      /* [N_USERS × N_ITEMS] – 0 = unrated           */
static float *user_mean_h;    /* [N_USERS]                                    */
static float *sim_matrix_h;   /* [N_USERS × N_USERS]                         */
static float *predictions_h;  /* [N_USERS × N_ITEMS]                         */

/* ── Device arrays ───────────────────────────────────────────────────────── */
static float *ratings_d;      /* [N_USERS × N_ITEMS]                         */
static float *user_mean_d;    /* [N_USERS]                                    */
static float *sim_matrix_d;   /* [N_USERS × N_USERS]                         */
static float *predictions_d;  /* [N_USERS × N_ITEMS]                         */

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set_h;
static int        test_size;

/* ── Convenience macros ──────────────────────────────────────────────────── */
#define R_H(u,i)    ratings_h[(u)*N_ITEMS + (i)]
#define SIM_H(u,v)  sim_matrix_h[(u)*N_USERS + (v)]
#define PRED_H(u,i) predictions_h[(u)*N_ITEMS + (i)]

/* ── CUDA utilities ──────────────────────────────────────────────────────── */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* ── Timing ──────────────────────────────────────────────────────────────── */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Host: Allocate / free ───────────────────────────────────────────────── */
static void alloc_arrays(void)
{
    ratings_h     = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
    user_mean_h   = (float *)calloc(N_USERS,            sizeof(float));
    sim_matrix_h  = (float *)calloc(N_USERS * N_USERS,  sizeof(float));
    predictions_h = (float *)calloc(N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings_h || !user_mean_h || !sim_matrix_h || !predictions_h) {
        fprintf(stderr, "Error: host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate device memory */
    int size_ratings = N_USERS * N_ITEMS * sizeof(float);
    int size_users   = N_USERS * sizeof(float);
    int size_sim     = N_USERS * N_USERS * sizeof(float);

    CUDA_CHECK(cudaMalloc(&ratings_d,     size_ratings));
    CUDA_CHECK(cudaMalloc(&user_mean_d,   size_users));
    CUDA_CHECK(cudaMalloc(&sim_matrix_d,  size_sim));
    CUDA_CHECK(cudaMalloc(&predictions_d, size_ratings));
}

static void free_arrays(void)
{
    free(ratings_h);
    free(user_mean_h);
    free(sim_matrix_h);
    free(predictions_h);
    free(test_set_h);

    cudaFree(ratings_d);
    cudaFree(user_mean_d);
    cudaFree(sim_matrix_d);
    cudaFree(predictions_d);
}

/* ── Host: Data generation ───────────────────────────────────────────────── */
static void generate_data(void)
{
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

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ── Host: User means ────────────────────────────────────────────────────── */
static void compute_user_means_host(void)
{
    for (int u = 0; u < N_USERS; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R_H(u, i) != 0.0f) { sum += R_H(u, i); cnt++; }
        }
        user_mean_h[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
}

/* ── CUDA Kernel: Compute Pearson similarities ────────────────────────────── */
__global__ void kernel_pearson_similarities(
    const float *ratings, const float *user_means,
    float *sim_matrix, int N_USERS, int N_ITEMS)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= N_USERS || v >= N_USERS) return;

    if (u == v) {
        sim_matrix[u * N_USERS + v] = 1.0f;
        return;
    }

    if (u > v) {
        /* Copy symmetric value */
        sim_matrix[u * N_USERS + v] = sim_matrix[v * N_USERS + u];
        return;
    }

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
        sim_matrix[u * N_USERS + v] = 0.0f;
        return;
    }

    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) {
        sim_matrix[u * N_USERS + v] = 0.0f;
        return;
    }

    float s = (float)(num / denom);
    s = fmax(-1.0f, fmin(1.0f, s));
    sim_matrix[u * N_USERS + v] = s;
}

/* ── CUDA Kernel: Compute predictions –––––––––––––––––––––––––––––––––––––– */
__global__ void kernel_predictions(
    const float *ratings, const float *user_means, const float *sim_matrix,
    float *predictions, int N_USERS, int N_ITEMS, int TOP_K)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int item = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= N_USERS || item >= N_ITEMS) return;

    float rating = ratings[u * N_ITEMS + item];
    if (rating != 0.0f) {
        predictions[u * N_ITEMS + item] = rating;
        return;
    }

    /* Find top-k neighbors with ratings for this item */
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
            /* Find minimum and replace if needed */
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

/* ── Host: Evaluate MAE ──────────────────────────────────────────────────── */
static float evaluate_mae_host(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED_H(test_set_h[t].user, test_set_h[t].item) 
                    - test_set_h[t].rating);
    return (float)(err / test_size);
}

static double similarity_checksum_host(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM_H(u, v);
    return s;
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("=== Pearson Correlation Recommender – CUDA Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d | Block size: %d\n\n",
           N_USERS, N_ITEMS, TOP_K, BLOCK_SIZE);

    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    t0 = now_sec(); generate_data();                 t1 = now_sec();
    printf("[Timing] Data generation       : %.4f s\n", t1 - t0);

    t0 = now_sec(); compute_user_means_host();       t1 = now_sec();
    printf("[Timing] User mean compute     : %.4f s\n", t1 - t0);

    /* Copy data to device */
    t0 = now_sec();
    int size_ratings = N_USERS * N_ITEMS * sizeof(float);
    int size_users   = N_USERS * sizeof(float);
    int size_sim     = N_USERS * N_USERS * sizeof(float);

    CUDA_CHECK(cudaMemcpy(ratings_d,   ratings_h,   size_ratings, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(user_mean_d, user_mean_h, size_users,   cudaMemcpyHostToDevice));
    t1 = now_sec();
    printf("[Timing] Host-to-device copy   : %.4f s\n", t1 - t0);

    /* Compute similarities on GPU */
    t0 = now_sec();
    int grid_dim = (N_USERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_dim, grid_dim);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    kernel_pearson_similarities<<<grid, block>>>(
        ratings_d, user_mean_d, sim_matrix_d, N_USERS, N_ITEMS);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = now_sec();
    t_sim = t1 - t0;
    printf("[Timing] Similarity matrix GPU : %.4f s\n", t_sim);

    /* Compute predictions on GPU */
    t0 = now_sec();
    grid_dim = (N_USERS + 16 - 1) / 16;
    int grid_dim2 = (N_ITEMS + 16 - 1) / 16;
    dim3 grid_pred(grid_dim, grid_dim2);
    dim3 block_pred(16, 16);
    kernel_predictions<<<grid_pred, block_pred>>>(
        ratings_d, user_mean_d, sim_matrix_d, predictions_d,
        N_USERS, N_ITEMS, TOP_K);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = now_sec();
    t_pred = t1 - t0;
    printf("[Timing] Predictions GPU       : %.4f s\n", t_pred);

    /* Copy results back to host */
    t0 = now_sec();
    CUDA_CHECK(cudaMemcpy(sim_matrix_h,  sim_matrix_d,  size_sim,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(predictions_h, predictions_d, size_ratings, cudaMemcpyDeviceToHost));
    t1 = now_sec();
    printf("[Timing] Device-to-host copy   : %.4f s\n", t1 - t0);

    /* Evaluation */
    float mae = evaluate_mae_host();
    double checksum = similarity_checksum_host();

    printf("\n[Results]\n");
    printf("  MAE on test set              : %.6f\n", mae);
    printf("  Sim-matrix checksum          : %.6f\n", checksum);
    printf("  Total (sim+pred)             : %.4f s\n", t_sim + t_pred);

    free_arrays();
    return EXIT_SUCCESS;
}
