#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/*
 * Hybrid MPI+OpenMP user-based collaborative filtering recommender.
 *
 * This version keeps the same recommendation algorithm as the serial baseline:
 * sparse user-item ratings, Pearson user-user similarity, and Top-K neighbor
 * prediction.  The difference is the execution model:
 *
 *   - MPI splits the user rows across processes.
 *   - OpenMP threads inside each process compute that process's rows.
 *   - MPI collectives share the row results needed by every process.
 *
 * The full ratings matrix is available on every rank because each rank uses the
 * same random seed to generate identical synthetic data.  Computed rows of
 * user_mean, sim_matrix, and predictions are then gathered or reduced as needed.
 */

#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

static int N_USERS;
static int N_ITEMS;
static int mpi_rank;
static int mpi_size;

static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* Matrices are stored as one-dimensional row-major arrays.  A zero in ratings means "unknown rating", not a real score. */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

static inline double now_sec(void) { return MPI_Wtime(); }

static void get_range(int rank, int size, int *start, int *end)
{
    /* Assign each MPI rank a contiguous block of user rows.  The last rank may
     * receive fewer rows when N_USERS is not divisible by the number of ranks. */
    int chunk = (N_USERS + size - 1) / size;
    *start = rank * chunk;
    *end   = *start + chunk;
    if (*end > N_USERS) *end = N_USERS;
}

static void alloc_arrays(void)
{
    ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        fprintf(stderr, "Rank %d: allocation failed.\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static void free_arrays(void)
{
    free(ratings); free(user_mean);
    free(sim_matrix); free(predictions);
    free(test_set);
}

static void generate_data(void)
{
    /* Every rank uses the same seed, so all ranks independently create the same
     * ratings matrix and the same held-out test set without broadcasting them. */
    srand(SEED);

    /* Approximate capacity for held-out ratings.  The actual count depends on
     * random sparsity and TEST_RATIO, so a small margin is added. */
    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            /* Skip most user-item pairs to model sparse recommendation data. */
            if ((float)rand() / RAND_MAX < SPARSITY) continue;
            float rating = (float)(rand() % 5) + 1.0f;
            /* Test ratings are hidden from training and used later only for
             * MAE/RMSE evaluation. */
            if ((float)rand() / RAND_MAX < TEST_RATIO && test_size < capacity) {
                test_set[test_size].user   = u;
                test_set[test_size].item   = i;
                test_set[test_size].rating = rating;
                test_size++;
            } else {
                R(u, i) = rating;
            }
        }
    }

    if (mpi_rank == 0)
        printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
               "Test ratings: %d\n",
               N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

static void compute_user_means(int start_u, int end_u,
                                int *recvcounts, int *displs)
{
    /* Each rank computes means only for its own user rows.  OpenMP distributes
     * those rows across threads inside the process. */
    #pragma omp parallel for schedule(static)
    for (int u = start_u; u < end_u; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        /* Use the middle of the 1..5 scale as a neutral fallback for users with
         * no observed training ratings. */
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }

    /* Pearson similarity needs every user's mean, so gather all rank-owned
     * mean segments into a complete user_mean array on every rank. */
    MPI_Allgatherv(
        &user_mean[start_u], end_u - start_u, MPI_FLOAT,
        user_mean, recvcounts, displs, MPI_FLOAT,
        MPI_COMM_WORLD
    );
}

static float pearson_similarity(int u, int v)
{
    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int    co  = 0;
    float  mu  = user_mean[u], mv = user_mean[v];

    for (int i = 0; i < N_ITEMS; i++) {
        if (R(u, i) != 0.0f && R(v, i) != 0.0f) {
            /* Pearson correlation compares centered ratings on items both users
             * rated, so user bias is removed before similarity is measured. */
            double du = R(u, i) - mu;
            double dv = R(v, i) - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    /* Require at least two co-rated items and nonzero variance; otherwise the
     * correlation is too weak or undefined. */
    if (co < 2) return 0.0f;
    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) return 0.0f;

    float s = (float)(num / denom);
    /* Clamp small numerical overshoots back to Pearson's valid range. */
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return s;
}

static void compute_all_similarities(int start_u, int end_u,
                                      int *recvcounts, int *displs)
{
    /* Each rank computes full similarity rows for its assigned users.  Dynamic
     * scheduling helps because different user pairs can have different numbers
     * of co-rated items. */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int u = start_u; u < end_u; u++) {
        SIM(u, u) = 1.0f;
        for (int v = 0; v < N_USERS; v++) {
            if (v == u) continue;
            SIM(u, v) = pearson_similarity(u, v);
        }
    }

    /* Predictions need similarities from all users, so gather every rank's
     * completed similarity rows into the full matrix on every rank. */
    MPI_Allgatherv(
        &sim_matrix[(size_t)start_u * N_USERS],
        (end_u - start_u) * N_USERS, MPI_FLOAT,
        sim_matrix, recvcounts, displs, MPI_FLOAT,
        MPI_COMM_WORLD
    );
}

typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    /* Sort candidate neighbors from most similar to least similar. */
    return (fb > fa) - (fb < fa);
}

static void compute_all_predictions(int start_u, int end_u)
{
    /* Prediction rows are independent, so each MPI rank handles only its user
     * block and OpenMP threads share those rows. */
    #pragma omp parallel
    {
        /* Allocate a private neighbor buffer per thread so qsort and writes to
         * nbrs do not race between OpenMP threads. */
        SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
        if (!nbrs) {
            fprintf(stderr, "Rank %d thread %d: malloc failed.\n",
                    mpi_rank, omp_get_thread_num());
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        #pragma omp for schedule(dynamic, 2)
        for (int u = start_u; u < end_u; u++) {
            for (int item = 0; item < N_ITEMS; item++) {

                /* Keep known training ratings unchanged; estimate only missing
                 * values. */
                if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

                int cnt = 0;
                for (int v = 0; v < N_USERS; v++) {
                    if (v == u || R(v, item) == 0.0f) continue;
                    /* Use only positively correlated users who rated this item
                     * as neighbors for the prediction. */
                    float s = SIM(u, v);
                    if (s <= 0.0f) continue;
                    nbrs[cnt].idx = v;
                    nbrs[cnt].val = s;
                    cnt++;
                }

                if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

                qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
                int k = (cnt < TOP_K) ? cnt : TOP_K;

                /* Mean-centered Top-K prediction:
                 *   prediction = user_mean +
                 *                weighted average of neighbor deviations
                 * This uses how much each neighbor liked the item relative to
                 * that neighbor's normal rating behavior. */
                double num = 0.0, den = 0.0;
                for (int j = 0; j < k; j++) {
                    float s = nbrs[j].val;
                    num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                    den += s;
                }

                float pred = (den > 1e-10)
                             ? user_mean[u] + (float)(num / den)
                             : user_mean[u];
                /* Keep estimates inside the synthetic 1..5 rating scale. */
                if (pred < 1.0f) pred = 1.0f;
                if (pred > 5.0f) pred = 5.0f;
                PRED(u, item) = pred;
            }
        }

        free(nbrs);
    }
}

static float evaluate_mae(int start_u, int end_u)
{
    double local_err = 0.0;
    int    local_cnt = 0;

    /* Each rank evaluates only test ratings whose user row it predicted. */
    #pragma omp parallel for reduction(+:local_err, local_cnt) schedule(static)
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        if (u < start_u || u >= end_u) continue;
        local_err += fabs(PRED(u, test_set[t].item) - test_set[t].rating);
        local_cnt++;
    }

    double global_err = 0.0;
    int    global_cnt = 0;

    /* Sum partial errors and counts on rank 0 to produce one global MAE. */
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_cnt > 0)
        return (float)(global_err / global_cnt);
    return 0.0f;
}

static float evaluate_rmse(int start_u, int end_u)
{
    double local_sq  = 0.0;
    int    local_cnt = 0;

    /* RMSE follows the same distributed pattern as MAE, but accumulates squared
     * prediction errors before the final square root. */
    #pragma omp parallel for reduction(+:local_sq, local_cnt) schedule(static)
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        if (u < start_u || u >= end_u) continue;
        double d = PRED(u, test_set[t].item) - test_set[t].rating;
        local_sq += d * d;
        local_cnt++;
    }

    double global_sq  = 0.0;
    int    global_cnt = 0;

    MPI_Reduce(&local_sq,  &global_sq,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_cnt > 0)
        return (float)sqrt(global_sq / global_cnt);
    return 0.0f;
}

static double similarity_checksum(void)
{
    /* Reproducibility check used to compare this implementation with serial and
     * other parallel versions. */
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

static void dump_test_predictions_json_mpi(int start_u, int end_u)
{
    /* Optional benchmark output.  When PRED_DUMP_PATH is set, gather all
     * prediction rows so rank 0 can write test predictions in serial order. */
    const char *path = getenv("PRED_DUMP_PATH");
    if (!path) return;

    int *pred_recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *pred_displs     = (int *)malloc(mpi_size * sizeof(int));
    for (int r = 0; r < mpi_size; r++) {
        int s, e;
        get_range(r, mpi_size, &s, &e);
        pred_recvcounts[r] = (e - s) * N_ITEMS;
        pred_displs[r]     = s * N_ITEMS;
    }
    MPI_Allgatherv(
        &predictions[(size_t)start_u * N_ITEMS],
        (end_u - start_u) * N_ITEMS,
        MPI_FLOAT,
        predictions,
        pred_recvcounts, pred_displs, MPI_FLOAT,
        MPI_COMM_WORLD
    );
    free(pred_recvcounts);
    free(pred_displs);

    if (mpi_rank != 0) return;

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[JSON]   Could not open %s for writing\n", path);
        return;
    }
    fputc('[', fp);
    for (int t = 0; t < test_size; t++) {
        if (t > 0) fputc(',', fp);
        fprintf(fp, "%.17g",
                (double)PRED(test_set[t].user, test_set[t].item));
    }
    fputs("]\n", fp);
    fclose(fp);
    printf("[JSON]   Test predictions written to %s (%d values)\n",
           path, test_size);
}

int main(int argc, char *argv[])
{
    int provided;
    /* FUNNELED means OpenMP worker threads do computation, while MPI calls are
     * made by the main thread outside parallel regions. */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED && mpi_rank == 0)
        fprintf(stderr, "Warning: MPI thread support below FUNNELED (%d < %d).\n",
                provided, MPI_THREAD_FUNNELED);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int start_u, end_u;
    get_range(mpi_rank, mpi_size, &start_u, &end_u);

    int omp_threads = omp_get_max_threads();

    if (mpi_rank == 0) {
        printf("=== Pearson Correlation Recommender – Hybrid MPI+OpenMP Version ===\n");
        printf("    Users: %d | Items: %d | Top-K: %d\n", N_USERS, N_ITEMS, TOP_K);
        printf("    MPI Processes: %d | OpenMP Threads/Process: %d"
               " | Total workers: %d\n\n",
               mpi_size, omp_threads, mpi_size * omp_threads);
    }

    int *mean_recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *mean_displs     = (int *)malloc(mpi_size * sizeof(int));
    int *sim_recvcounts  = (int *)malloc(mpi_size * sizeof(int));
    int *sim_displs      = (int *)malloc(mpi_size * sizeof(int));

    /* Build MPI_Allgatherv metadata for row-block data.  Mean arrays gather one
     * float per user; similarity rows gather N_USERS floats per local user. */
    for (int r = 0; r < mpi_size; r++) {
        int s, e;
        get_range(r, mpi_size, &s, &e);
        int local_rows      = e - s;
        mean_recvcounts[r]  = local_rows;
        mean_displs[r]      = s;
        sim_recvcounts[r]   = local_rows * N_USERS;
        sim_displs[r]       = s * N_USERS;
    }

    alloc_arrays();

    double t0, t1, t_sim, t_pred;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    generate_data();
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_user_means(start_u, end_u, mean_recvcounts, mean_displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] User mean compute  : %.4f s"
               "  [MPI %d × OMP %d]\n", t1 - t0, mpi_size, omp_threads);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_all_similarities(start_u, end_u, sim_recvcounts, sim_displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t_sim = now_sec() - t0;
    if (mpi_rank == 0) {
        printf("[Timing] Similarity matrix  : %.4f s"
               "  [MPI %d × OMP %d]\n", t_sim, mpi_size, omp_threads);
        printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_all_predictions(start_u, end_u);
    MPI_Barrier(MPI_COMM_WORLD);
    t_pred = now_sec() - t0;
    if (mpi_rank == 0)
        printf("[Timing] Prediction phase   : %.4f s"
               "  [MPI %d × OMP %d]\n", t_pred, mpi_size, omp_threads);

    dump_test_predictions_json_mpi(start_u, end_u);

    float mae  = evaluate_mae(start_u, end_u);
    float rmse = evaluate_rmse(start_u, end_u);
    if (mpi_rank == 0) {
        printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
               mae, test_size);
        printf("[Eval]   RMSE on test set   : %.4f  (test size: %d)\n",
               rmse, test_size);
        printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);
    }

    if (mpi_rank == 0) {
        int show_u = (N_USERS < 5) ? N_USERS : 5;
        int show_i = (N_ITEMS < 5) ? N_ITEMS : 5;
        printf("\n--- Sample Predictions (first %d users, %d items) ---\n",
               show_u, show_i);
        printf("%-9s", "User\\Item");
        for (int i = 0; i < show_i; i++) printf("  Item%-3d", i);
        printf("\n");
        for (int u = 0; u < show_u; u++) {
            printf("User %-4d", u);
            for (int i = 0; i < show_i; i++) printf("  %5.2f  ", PRED(u, i));
            printf("\n");
        }
    }

    free(mean_recvcounts); free(mean_displs);
    free(sim_recvcounts);  free(sim_displs);
    free_arrays();

    MPI_Finalize();
    return 0;
}
