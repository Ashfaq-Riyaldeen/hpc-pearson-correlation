/* ============================================================================
 *  Pearson Correlation Recommender – Hybrid MPI + OpenMP version.
 *
 *  Same algorithm as serial_recommender.c, but parallelized two ways:
 *
 *    MPI (between processes):
 *      The N_USERS rows are split into contiguous slabs; each rank owns one
 *      slab and only computes user_mean / similarity / predictions for its
 *      own users. After each phase, results are gathered to all ranks with
 *      MPI_Allgatherv so every rank has the data it needs for the next phase.
 *
 *    OpenMP (inside each process):
 *      Within a rank's slab, the outer "for each owned user" loop is split
 *      across threads with `#pragma omp parallel for`. Schedules are tuned
 *      per phase (static for balanced loops, dynamic for irregular work).
 *
 *  Every rank holds the FULL N_USERS x N_ITEMS / N_USERS x N_USERS buffers;
 *  this keeps indexing identical to the serial code at the cost of memory.
 *  For very large problems a distributed layout would be required.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/* Same tuning constants as the serial reference — see serial_recommender.c. */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

static int N_USERS;
static int N_ITEMS;
static int mpi_rank;   /* This process's MPI rank (0 .. mpi_size-1). */
static int mpi_size;   /* Total number of MPI processes. */

/* Full-size buffers — every rank holds the entire matrix even though it only
 * WRITES to its assigned slice. After each phase Allgatherv fills in the
 * slices owned by other ranks. */
static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

/* MPI_Wtime is a monotonic, high-resolution timer suitable for measuring
 * parallel phases — consistent across ranks. */
static inline double now_sec(void) { return MPI_Wtime(); }

/* Compute the [start, end) row range this rank owns.
 *
 * We use a "ceiling-divide" chunk so that — except for the last rank, which
 * may be slightly smaller — every rank gets the same number of rows. This
 * gives a contiguous, gap-free partition that maps cleanly onto the
 * Allgatherv displacement arrays built in main(). */
static void get_range(int rank, int size, int *start, int *end)
{
    int chunk = (N_USERS + size - 1) / size;
    *start = rank * chunk;
    *end   = *start + chunk;
    if (*end > N_USERS) *end = N_USERS;
}

/* Allocate full-size buffers on every rank. calloc() zeros them, which is
 * critical: 0.0f in `ratings` means "no rating", and the Allgatherv phases
 * later need uninitialized slices to start at 0 so partial sums never leak. */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        /* MPI_Abort tears down ALL ranks — a failure on one rank means the
         * collective communications later would deadlock anyway. */
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

/* Build the synthetic dataset.
 *
 * IMPORTANT: every rank runs this independently with the SAME fixed SEED, so
 * each rank ends up with byte-identical `ratings` and `test_set` arrays.
 * This is the cheapest way to keep the data consistent across ranks (no
 * broadcast needed) and only works because the RNG is deterministic. */
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
                R(u, i) = rating;
            }
        }
    }

    if (mpi_rank == 0)
        printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
               "Test ratings: %d\n",
               N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* Compute per-user mean ratings, parallelized at TWO levels:
 *   MPI:    this rank handles users [start_u, end_u) only.
 *   OpenMP: `parallel for schedule(static)` — each user does the same amount
 *           of work (a scan over N_ITEMS), so static balances threads well.
 *
 * After the local work, MPI_Allgatherv stitches every rank's slice of
 * user_mean back together so all ranks have the full vector ready for the
 * similarity phase. */
static void compute_user_means(int start_u, int end_u,
                                int *recvcounts, int *displs)
{
    #pragma omp parallel for schedule(static)
    for (int u = start_u; u < end_u; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }

    /* Allgatherv: each rank contributes its slice, every rank receives the
     * full concatenated vector. `recvcounts`/`displs` encode the partition. */
    MPI_Allgatherv(
        &user_mean[start_u], end_u - start_u, MPI_FLOAT,
        user_mean, recvcounts, displs, MPI_FLOAT,
        MPI_COMM_WORLD
    );
}

/* Pearson correlation between two users over their co-rated items.
 * Identical math to the serial version — see serial_recommender.c for the
 * derivation and edge-case notes. This function is THREAD-SAFE: it only
 * reads from the shared `ratings`/`user_mean` arrays, so many OpenMP
 * threads can call it concurrently with no synchronization. */
static float pearson_similarity(int u, int v)
{
    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int    co  = 0;
    float  mu  = user_mean[u], mv = user_mean[v];

    for (int i = 0; i < N_ITEMS; i++) {
        if (R(u, i) != 0.0f && R(v, i) != 0.0f) {
            double du = R(u, i) - mu;
            double dv = R(v, i) - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    if (co < 2) return 0.0f;
    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) return 0.0f;

    float s = (float)(num / denom);
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return s;
}

/* Build the user-user similarity matrix in parallel.
 *
 * Note: unlike the serial version, this implementation does NOT exploit
 * symmetry (it computes both SIM(u,v) and SIM(v,u) independently). The
 * reason is data ownership — rank R only owns its row block, so it cannot
 * write into rows owned by other ranks. Recomputing both halves keeps the
 * MPI partition simple at the cost of 2x similarity work.
 *
 * OpenMP schedule(dynamic, 4): rows differ in cost because some users have
 * many more rated items (and thus heavier inner loops) than others. Dynamic
 * scheduling lets idle threads pick up new row chunks as they finish. */
static void compute_all_similarities(int start_u, int end_u,
                                      int *recvcounts, int *displs)
{
    #pragma omp parallel for schedule(dynamic, 4)
    for (int u = start_u; u < end_u; u++) {
        SIM(u, u) = 1.0f;
        for (int v = 0; v < N_USERS; v++) {
            if (v == u) continue;
            SIM(u, v) = pearson_similarity(u, v);
        }
    }

    /* Stitch every rank's contiguous block of rows back into the full
     * N_USERS x N_USERS matrix on all ranks. */
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
    return (fb > fa) - (fb < fa);
}

/* Generate predictions for this rank's user slab.
 *
 * Same Top-K weighted-average algorithm as the serial version. Two
 * parallelization details worth highlighting:
 *
 *   1. The `nbrs` scratch array is allocated INSIDE `#pragma omp parallel`,
 *      so each thread gets its own private buffer. Sharing a single buffer
 *      between threads would race on the qsort and on the index `cnt`.
 *
 *   2. schedule(dynamic, 2): the per-user cost is highly irregular (a user
 *      with many unknown items does a lot of qsorts; one with few does
 *      almost none), so dynamic scheduling beats static here.
 *
 * Notably there is NO MPI communication at the end of this function — each
 * rank just writes into its own slice of `predictions`. The slices that
 * other ranks own remain zero on this rank. That's fine because evaluation
 * uses a parallel reduction (see evaluate_mae / evaluate_rmse). */
static void compute_all_predictions(int start_u, int end_u)
{
    #pragma omp parallel
    {
        /* Per-thread scratch buffer — see comment above the function. */
        SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
        if (!nbrs) {
            fprintf(stderr, "Rank %d thread %d: malloc failed.\n",
                    mpi_rank, omp_get_thread_num());
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        #pragma omp for schedule(dynamic, 2)
        for (int u = start_u; u < end_u; u++) {
            for (int item = 0; item < N_ITEMS; item++) {

                /* Already-rated cell: keep the observed rating. */
                if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

                /* Collect candidate neighbors: rated this item & positively similar. */
                int cnt = 0;
                for (int v = 0; v < N_USERS; v++) {
                    if (v == u || R(v, item) == 0.0f) continue;
                    float s = SIM(u, v);
                    if (s <= 0.0f) continue;
                    nbrs[cnt].idx = v;
                    nbrs[cnt].val = s;
                    cnt++;
                }

                /* Cold start for this cell -> fall back to user mean. */
                if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

                /* Rank by similarity and keep the K best. */
                qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
                int k = (cnt < TOP_K) ? cnt : TOP_K;

                double num = 0.0, den = 0.0;
                for (int j = 0; j < k; j++) {
                    float s = nbrs[j].val;
                    num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                    den += s;
                }

                float pred = (den > 1e-10)
                             ? user_mean[u] + (float)(num / den)
                             : user_mean[u];
                /* Clamp to the valid 1..5 rating range. */
                if (pred < 1.0f) pred = 1.0f;
                if (pred > 5.0f) pred = 5.0f;
                PRED(u, item) = pred;
            }
        }

        free(nbrs);
    }
}

/* Mean Absolute Error, computed in three levels:
 *   - each thread accumulates its own partial via OpenMP reduction(+:...),
 *   - the rank-local sum covers only test entries whose user lies in this
 *     rank's slab (since other ranks didn't fill in those predictions),
 *   - MPI_Reduce sums across ranks; rank 0 divides to get the final MAE. */
static float evaluate_mae(int start_u, int end_u)
{
    double local_err = 0.0;
    int    local_cnt = 0;

    #pragma omp parallel for reduction(+:local_err, local_cnt) schedule(static)
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        /* Skip test entries owned by other ranks — their PRED is still 0 here. */
        if (u < start_u || u >= end_u) continue;
        local_err += fabs(PRED(u, test_set[t].item) - test_set[t].rating);
        local_cnt++;
    }

    double global_err = 0.0;
    int    global_cnt = 0;

    /* Sum partials across all ranks; only rank 0 needs the result. */
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_cnt > 0)
        return (float)(global_err / global_cnt);
    return 0.0f;
}

/* Root Mean Squared Error — same reduction pattern as evaluate_mae(), but
 * accumulates squared errors and takes a square root at the end. */
static float evaluate_rmse(int start_u, int end_u)
{
    double local_sq  = 0.0;
    int    local_cnt = 0;

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

/* Whole-matrix similarity checksum.
 * After the Allgatherv in compute_all_similarities(), every rank holds the
 * full matrix — so the sum is identical on each rank and we don't need an
 * MPI reduction here. Threaded with OpenMP reduction for speed. */
static double similarity_checksum(void)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

int main(int argc, char *argv[])
{
    /* MPI_THREAD_FUNNELED is the minimum threading mode we need: it allows
     * OpenMP threads to run, but guarantees that only the main thread of
     * each process makes MPI calls. That matches our design — every MPI
     * call below sits OUTSIDE any `#pragma omp parallel` region. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED && mpi_rank == 0)
        fprintf(stderr, "Warning: MPI thread support below FUNNELED (%d < %d).\n",
                provided, MPI_THREAD_FUNNELED);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    /* CLI overrides: mpirun -np N ./hybrid_recommender [num_users] [num_items]. */
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Figure out which slice of users THIS rank is responsible for. */
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

    /* Pre-compute the recvcounts/displs arrays used by MPI_Allgatherv.
     *
     * Allgatherv needs to know, for each contributing rank, how many
     * elements it sends and where those elements live in the destination
     * buffer. Two sets here:
     *   mean_*  – one float per user        (N_USERS elements total)
     *   sim_*   – one row of N_USERS floats per user (N_USERS^2 total)
     *
     * Building these once up front avoids re-computing them every phase. */
    int *mean_recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *mean_displs     = (int *)malloc(mpi_size * sizeof(int));
    int *sim_recvcounts  = (int *)malloc(mpi_size * sizeof(int));
    int *sim_displs      = (int *)malloc(mpi_size * sizeof(int));

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

    /* Each phase is wrapped in MPI_Barrier()...timer...MPI_Barrier() so the
     * reported timing reflects the slowest rank — the realistic wall-clock
     * cost of the phase across the whole job, not a fast rank's local time. */
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
