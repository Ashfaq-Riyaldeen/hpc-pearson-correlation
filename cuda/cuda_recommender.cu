#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

static int N_USERS;
static int N_ITEMS;

static float *h_ratings;
static float *h_user_mean;
static float *h_sim_matrix;
static float *h_predictions;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                    \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

__global__ void kernel_user_means(const float *ratings, float *user_mean,
                                   int N_USERS, int N_ITEMS)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= N_USERS) return;

    double sum = 0.0;
    int    cnt = 0;
    for (int i = 0; i < N_ITEMS; i++) {
        float r = ratings[u * N_ITEMS + i];
        if (r != 0.0f) { sum += r; cnt++; }
    }

    user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
}

__global__ void kernel_similarity(const float *ratings, const float *user_mean,
                                   float *sim_matrix, int N_USERS, int N_ITEMS)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= N_USERS || v >= N_USERS) return;

    if (u == v) { sim_matrix[u * N_USERS + v] = 1.0f; return; }

    if (u > v) return;

    float  mu = user_mean[u], mv = user_mean[v];
    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int    co  = 0;

    for (int i = 0; i < N_ITEMS; i++) {
        float ru = ratings[u * N_ITEMS + i];
        float rv = ratings[v * N_ITEMS + i];

        if (ru != 0.0f && rv != 0.0f) {
            double du = (double)ru - mu;
            double dv = (double)rv - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    float s = 0.0f;
    if (co >= 2) {
        double denom = sqrt(den_u) * sqrt(den_v);
        if (denom >= 1e-10) {
            s = (float)(num / denom);
            if (s >  1.0f) s =  1.0f;
            if (s < -1.0f) s = -1.0f;
        }
    }

    sim_matrix[u * N_USERS + v] = s;
    sim_matrix[v * N_USERS + u] = s;
}

__global__ void kernel_predictions(const float *ratings, const float *user_mean,
                                    const float *sim_matrix, float *predictions,
                                    int N_USERS, int N_ITEMS)
{
    int u    = blockIdx.x * blockDim.x + threadIdx.x;
    int item = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= N_USERS || item >= N_ITEMS) return;

    float r_ui = ratings[u * N_ITEMS + item];
    if (r_ui != 0.0f) {
        predictions[u * N_ITEMS + item] = r_ui;
        return;
    }

    float top_sim[TOP_K];
    float top_rat[TOP_K];
    float top_mu [TOP_K];
    int   top_cnt = 0;
    float worst   = -2.0f;
    int   worst_j = 0;

    for (int v = 0; v < N_USERS; v++) {
        if (v == u) continue;

        float rv = ratings[v * N_ITEMS + item];
        if (rv == 0.0f) continue;

        float s = sim_matrix[u * N_USERS + v];
        if (s <= 0.0f) continue;

        if (top_cnt < TOP_K) {
            top_sim[top_cnt] = s;
            top_rat[top_cnt] = rv;
            top_mu [top_cnt] = user_mean[v];
            if (top_cnt == 0 || s < worst) { worst = s; worst_j = top_cnt; }
            top_cnt++;
        } else if (s > worst) {
            top_sim[worst_j] = s;
            top_rat[worst_j] = rv;
            top_mu [worst_j] = user_mean[v];

            worst = top_sim[0]; worst_j = 0;
            for (int j = 1; j < TOP_K; j++) {
                if (top_sim[j] < worst) { worst = top_sim[j]; worst_j = j; }
            }
        }
    }

    float mu_u = user_mean[u];
    if (top_cnt == 0) { predictions[u * N_ITEMS + item] = mu_u; return; }

    double num = 0.0, den = 0.0;
    for (int j = 0; j < top_cnt; j++) {
        double s = top_sim[j];
        num += s * ((double)top_rat[j] - top_mu[j]);
        den += s;
    }

    float pred = (den > 1e-10) ? mu_u + (float)(num / den) : mu_u;

    if (pred < 1.0f) pred = 1.0f;
    if (pred > 5.0f) pred = 5.0f;
    predictions[u * N_ITEMS + item] = pred;
}

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

static float evaluate_rmse(void)
{
    if (test_size == 0) return 0.0f;
    double sq = 0.0;
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user, i = test_set[t].item;
        double d = h_predictions[u * N_ITEMS + i] - test_set[t].rating;
        sq += d * d;
    }
    return (float)sqrt(sq / test_size);
}

static void dump_test_predictions_json(const char *path)
{
    if (!path) return;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[JSON]   Could not open %s for writing\n", path);
        return;
    }
    fputc('[', fp);
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user, i = test_set[t].item;
        if (t > 0) fputc(',', fp);
        fprintf(fp, "%.17g", (double)h_predictions[u * N_ITEMS + i]);
    }
    fputs("]\n", fp);
    fclose(fp);
    printf("[JSON]   Test predictions written to %s (%d values)\n",
           path, test_size);
}

static double similarity_checksum(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += h_sim_matrix[u * N_USERS + v];
    return s;
}

int main(int argc, char *argv[])
{
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== Pearson Correlation Recommender – CUDA Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d\n", N_USERS, N_ITEMS, TOP_K);
    printf("    GPU: %s  |  SMs: %d  |  Compute: %d.%d\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    alloc_host();

    double t0, t1;

    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    float *d_ratings, *d_user_mean, *d_sim, *d_pred;
    CUDA_CHECK(cudaMalloc(&d_ratings,  (size_t)N_USERS * N_ITEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_user_mean, (size_t)N_USERS           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sim,       (size_t)N_USERS * N_USERS  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred,      (size_t)N_USERS * N_ITEMS  * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_sim,  0, (size_t)N_USERS * N_USERS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pred, 0, (size_t)N_USERS * N_ITEMS * sizeof(float)));

    t0 = now_sec();
    CUDA_CHECK(cudaMemcpy(d_ratings, h_ratings,
                          (size_t)N_USERS * N_ITEMS * sizeof(float),
                          cudaMemcpyHostToDevice));
    t1 = now_sec();
    printf("[Timing] H→D ratings        : %.4f s\n", t1 - t0);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    float ms;

    {
        int block = 256;
        int grid  = (N_USERS + block - 1) / block;

        CUDA_CHECK(cudaEventRecord(ev_start));
        kernel_user_means<<<grid, block>>>(d_ratings, d_user_mean,
                                           N_USERS, N_ITEMS);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        printf("[Timing] User mean compute  : %.4f s  [GPU kernel]\n",
               ms / 1000.0f);

        CUDA_CHECK(cudaMemcpy(h_user_mean, d_user_mean,
                              N_USERS * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    double t_sim;
    {
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

        CUDA_CHECK(cudaMemcpy(h_sim_matrix, d_sim,
                              (size_t)N_USERS * N_USERS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());
    }

    double t_pred;
    {
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

        CUDA_CHECK(cudaMemcpy(h_predictions, d_pred,
                              (size_t)N_USERS * N_ITEMS * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    dump_test_predictions_json(getenv("PRED_DUMP_PATH"));

    printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
           evaluate_mae(), test_size);
    printf("[Eval]   RMSE on test set   : %.4f  (test size: %d)\n",
           evaluate_rmse(), test_size);
    printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);

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

    CUDA_CHECK(cudaFree(d_ratings));
    CUDA_CHECK(cudaFree(d_user_mean));
    CUDA_CHECK(cudaFree(d_sim));
    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free_host();
    return 0;
}
