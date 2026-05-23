#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <mpi.h>

/*
 * Hybrid MPI+Pthreads user-based collaborative filtering recommender.
 *
 * The recommendation algorithm matches the serial baseline: generate sparse
 * ratings, hold out a test set, compute Pearson user-user similarities, and
 * predict missing ratings from the TOP_K most similar positive neighbors.
 *
 * Parallel model:
 *   1. MPI assigns each process a contiguous block of user rows.
 *   2. Pthreads inside each process split that rank-owned row block.
 *   3. MPI gathers distributed user means and similarity rows when later phases
 *      need the complete arrays on every rank.
 *   4. MPI reductions combine local MAE/RMSE errors into rank 0 results.
 */

#define DEFAULT_USERS    1000
#define DEFAULT_ITEMS    1000
#define DEFAULT_THREADS     4
#define MAX_THREADS        64
#define SPARSITY         0.70f
#define TOP_K              20
#define SEED               42
#define TEST_RATIO        0.10f

static int N_USERS;
static int N_ITEMS;
static int N_THREADS;
static int mpi_rank;
static int mpi_size;

static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* Row-major matrix indexing helpers.  In ratings[], zero means "unknown";
 * generated ratings use the 1..5 scale. */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

typedef struct {
    int tid;
    int nthreads;
    /* User rows owned by this MPI rank.  Each pthread receives a subrange
     * inside [rank_start, rank_end). */
    int rank_start;
    int rank_end;
} ThreadArgs;

static inline double now_sec(void) { return MPI_Wtime(); }

/* MPI rank-level row partition (matches hybrid/mpi_openmp version). */
static void get_range(int rank, int size, int *start, int *end)
{
    /* MPI-level partition: rank r owns a contiguous block of user rows. */
    int chunk = (N_USERS + size - 1) / size;
    *start = rank * chunk;
    *end   = *start + chunk;
    if (*end > N_USERS) *end = N_USERS;
}

/* Pthread-level partition of [0, total) for tid in [0, nthreads). */
static void work_range(int total, int tid, int nthreads, int *start, int *end)
{
    /* Thread-level partition: split one rank's local rows among pthreads. */
    int chunk = (total + nthreads - 1) / nthreads;
    *start = tid * chunk;
    *end   = *start + chunk;
    if (*end > total) *end = total;
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
    /* Same seed on every rank means each process independently creates the same
     * synthetic ratings and test set without broadcasting them. */
    srand(SEED);

    /* Approximate capacity for held-out ratings; the exact count depends on the
     * random sparsity pattern and TEST_RATIO. */
    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            /* Skip most cells to simulate a sparse recommender dataset. */
            if ((float)rand() / RAND_MAX < SPARSITY) continue;
            float rating = (float)(rand() % 5) + 1.0f;
            /* Held-out test ratings are hidden from training and used only for
             * final prediction-error evaluation. */
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

/* ── Phase 2: per-thread user-mean slice of this rank's row range. ── */
static void *thread_user_means(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int rank_rows = a->rank_end - a->rank_start;
    int rel_start, rel_end;
    /* Convert this thread's relative slice of the rank block into absolute
     * user ids. */
    work_range(rank_rows, a->tid, a->nthreads, &rel_start, &rel_end);
    int start = a->rank_start + rel_start;
    int end   = a->rank_start + rel_end;

    for (int u = start; u < end; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        /* Neutral fallback for users with no observed training ratings. */
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
    return NULL;
}

static float pearson_similarity(int u, int v)
{
    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int    co  = 0;
    float  mu  = user_mean[u], mv = user_mean[v];

    for (int i = 0; i < N_ITEMS; i++) {
        if (R(u, i) != 0.0f && R(v, i) != 0.0f) {
            /* Pearson correlation uses centered ratings on co-rated items, so
             * each user's normal rating level is removed. */
            double du = R(u, i) - mu;
            double dv = R(v, i) - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    /* Correlation is unreliable with fewer than two shared ratings and
     * undefined when either user has no variance around their mean. */
    if (co < 2) return 0.0f;
    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) return 0.0f;

    float s = (float)(num / denom);
    /* Clamp small floating-point overshoots back into Pearson's valid range. */
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return s;
}

/* ── Phase 3: per-thread slice of this rank's similarity rows. ──
 * Mirrors the OpenMP hybrid's row-major fill (u in rank slice, v across
 * all users) so the resulting per-rank block is laid out exactly as the
 * MPI_Allgatherv expects and the similarity checksum matches serial bit
 * for bit. */
static void *thread_similarities(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int rank_rows = a->rank_end - a->rank_start;
    int rel_start, rel_end;
    /* Each pthread computes complete similarity rows for a subset of this
     * rank's users. */
    work_range(rank_rows, a->tid, a->nthreads, &rel_start, &rel_end);
    int start = a->rank_start + rel_start;
    int end   = a->rank_start + rel_end;

    for (int u = start; u < end; u++) {
        SIM(u, u) = 1.0f;
        for (int v = 0; v < N_USERS; v++) {
            if (v == u) continue;
            SIM(u, v) = pearson_similarity(u, v);
        }
    }
    return NULL;
}

typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    /* Sort neighbors from strongest similarity to weakest for Top-K selection. */
    return (fb > fa) - (fb < fa);
}

/* ── Phase 4: per-thread prediction slice within rank's row range. ──
 * Each pthread allocates its own private nbrs[] buffer so the TOP-K
 * neighbour selection is race-free without any mutex. */
static void *thread_predictions(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int rank_rows = a->rank_end - a->rank_start;
    int rel_start, rel_end;
    /* Prediction rows are independent, so the same row-splitting helper works
     * for this phase too. */
    work_range(rank_rows, a->tid, a->nthreads, &rel_start, &rel_end);
    int start = a->rank_start + rel_start;
    int end   = a->rank_start + rel_end;

    /* Private per-thread buffer avoids races while collecting and sorting
     * neighbor candidates. */
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
    if (!nbrs) {
        fprintf(stderr, "Rank %d thread %d: malloc failed.\n",
                mpi_rank, a->tid);
        return NULL;
    }

    for (int u = start; u < end; u++) {
        for (int item = 0; item < N_ITEMS; item++) {

            /* Known training ratings do not need estimation. */
            if (R(u, item) != 0.0f) {
                PRED(u, item) = R(u, item);
                continue;
            }

            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u || R(v, item) == 0.0f) continue;
                /* Use only positive neighbors that already rated this item. */
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
             *   pred(u,i) = mean(u) + weighted average of
             *               neighbor_rating(v,i) - mean(v)
             */
            double num = 0.0, den = 0.0;
            for (int j = 0; j < k; j++) {
                float s = nbrs[j].val;
                num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                den += s;
            }

            float pred = (den > 1e-10)
                         ? user_mean[u] + (float)(num / den)
                         : user_mean[u];
            /* Keep predicted ratings on the 1..5 scale. */
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
    return NULL;
}

/* ── Pthread launch/join harness; runs on the main MPI thread only. ── */
static double run_parallel(void *(*fn)(void *),
                           pthread_t  *threads,
                           ThreadArgs *args,
                           int rank_start, int rank_end)
{
    double t0 = now_sec();

    /* Create workers for one phase.  After all threads join, the main thread is
     * free to perform any MPI collective required by that phase. */
    for (int t = 0; t < N_THREADS; t++) {
        args[t].tid        = t;
        args[t].nthreads   = N_THREADS;
        args[t].rank_start = rank_start;
        args[t].rank_end   = rank_end;
        if (pthread_create(&threads[t], NULL, fn, &args[t]) != 0) {
            fprintf(stderr, "Rank %d: pthread_create failed for thread %d\n",
                    mpi_rank, t);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    for (int t = 0; t < N_THREADS; t++)
        pthread_join(threads[t], NULL);

    return now_sec() - t0;
}

/* ── Phase 5: sequential per-rank reduction + MPI_Reduce to rank 0. ──
 * test_size is small (≈30k entries) so threading the filter loop adds
 * no measurable benefit; keep it sequential per rank. */
static float evaluate_mae(int start_u, int end_u)
{
    double local_err = 0.0;
    int    local_cnt = 0;
    /* Each rank evaluates only the held-out ratings for users it predicted. */
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        if (u < start_u || u >= end_u) continue;
        local_err += fabs(PRED(u, test_set[t].item) - test_set[t].rating);
        local_cnt++;
    }

    double global_err = 0.0;
    int    global_cnt = 0;
    /* Rank 0 receives the summed absolute error and total test count. */
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_cnt > 0)
        return (float)(global_err / global_cnt);
    return 0.0f;
}

static float evaluate_rmse(int start_u, int end_u)
{
    /* Same distributed evaluation as MAE, but with squared errors and a final
     * square root on rank 0. */
    double local_sq  = 0.0;
    int    local_cnt = 0;
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
    /* Simple reproducibility check against the serial and other parallel
     * versions. */
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

static void dump_test_predictions_json_mpi(int start_u, int end_u)
{
    /* Optional benchmark output.  When PRED_DUMP_PATH is set, gather all
     * distributed prediction rows so rank 0 can write test predictions in the
     * same order as the serial implementation. */
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
    /* FUNNELED is sufficient: pthread workers do local CPU work, while only the
     * main thread calls MPI routines. */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (provided < MPI_THREAD_FUNNELED && mpi_rank == 0)
        fprintf(stderr, "Warning: MPI thread support below FUNNELED (%d < %d).\n",
                provided, MPI_THREAD_FUNNELED);

    N_USERS   = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS   = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;
    N_THREADS = (argc >= 4) ? atoi(argv[3]) : DEFAULT_THREADS;

    if (N_USERS <= 0 || N_ITEMS <= 0 || N_THREADS <= 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s [num_users] [num_items] [num_threads]\n",
                    argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (N_THREADS > MAX_THREADS) {
        if (mpi_rank == 0)
            fprintf(stderr, "Warning: capping threads at %d\n", MAX_THREADS);
        N_THREADS = MAX_THREADS;
    }

    int start_u, end_u;
    get_range(mpi_rank, mpi_size, &start_u, &end_u);

    if (mpi_rank == 0) {
        printf("=== Pearson Correlation Recommender – Hybrid MPI+Pthreads Version ===\n");
        printf("    Users: %d | Items: %d | Top-K: %d\n", N_USERS, N_ITEMS, TOP_K);
        printf("    MPI Processes: %d | Pthreads/Process: %d"
               " | Total workers: %d\n\n",
               mpi_size, N_THREADS, mpi_size * N_THREADS);
    }

    int *mean_recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *mean_displs     = (int *)malloc(mpi_size * sizeof(int));
    int *sim_recvcounts  = (int *)malloc(mpi_size * sizeof(int));
    int *sim_displs      = (int *)malloc(mpi_size * sizeof(int));

    /* MPI_Allgatherv metadata for row-block data.  user_mean gathers one float
     * per local user; sim_matrix gathers N_USERS floats per local user row. */
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

    pthread_t  threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];

    double t0, t1, t_sim, t_pred;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    generate_data();
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* Phase 2: pthread workers compute local user means, then MPI gathers the
     * complete user_mean array onto every rank for Pearson similarity. */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    run_parallel(thread_user_means, threads, args, start_u, end_u);
    MPI_Allgatherv(
        &user_mean[start_u], end_u - start_u, MPI_FLOAT,
        user_mean, mean_recvcounts, mean_displs, MPI_FLOAT,
        MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] User mean compute  : %.4f s"
               "  [MPI %d × PT %d]\n", t1 - t0, mpi_size, N_THREADS);

    /* Phase 3: pthread workers compute this rank's similarity rows, then MPI
     * gathers all rows so every rank can make predictions independently. */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    run_parallel(thread_similarities, threads, args, start_u, end_u);
    MPI_Allgatherv(
        &sim_matrix[(size_t)start_u * N_USERS],
        (end_u - start_u) * N_USERS, MPI_FLOAT,
        sim_matrix, sim_recvcounts, sim_displs, MPI_FLOAT,
        MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t_sim = now_sec() - t0;
    if (mpi_rank == 0) {
        printf("[Timing] Similarity matrix  : %.4f s"
               "  [MPI %d × PT %d]\n", t_sim, mpi_size, N_THREADS);
        printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());
    }

    /* Phase 4: each rank now has the full SIM matrix; it predicts only its own
     * user rows, so no communication is needed during prediction. */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    run_parallel(thread_predictions, threads, args, start_u, end_u);
    MPI_Barrier(MPI_COMM_WORLD);
    t_pred = now_sec() - t0;
    if (mpi_rank == 0)
        printf("[Timing] Prediction phase   : %.4f s"
               "  [MPI %d × PT %d]\n", t_pred, mpi_size, N_THREADS);

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
