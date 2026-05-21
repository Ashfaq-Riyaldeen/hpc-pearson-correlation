/*
 * mpi_recommender.c
 * Pearson Correlation Recommender – MPI (Distributed Memory) Version
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  WHAT IS MPI?                                                       │
 * │                                                                     │
 * │  MPI (Message Passing Interface) is the industry standard for      │
 * │  distributed-memory parallel programming.  Unlike OpenMP/Pthreads  │
 * │  where threads share one memory space, every MPI "rank" (process)  │
 * │  has its OWN private address space.  Data cannot be read from      │
 * │  another process by dereferencing a pointer — it must be           │
 * │  explicitly sent and received using MPI calls.                     │
 * │                                                                     │
 * │  mpirun starts N identical copies of this executable.  Each copy  │
 * │  learns its own rank ID (0-based) and the total process count via  │
 * │  MPI_Comm_rank() and MPI_Comm_size().  All ranks then execute the  │
 * │  same main(), branching on rank where needed (e.g. only rank 0     │
 * │  prints results).                                                   │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  PARALLELISATION STRATEGY (row partitioning)                       │
 * │                                                                     │
 * │  N_USERS rows are divided evenly across all ranks.                 │
 * │  Rank r owns the users in the range [start_u, end_u).             │
 * │                                                                     │
 * │  Phase 1  Data gen      All ranks call srand(SEED) and run the    │
 * │                          same rand() sequence → identical data on  │
 * │                          every rank.  No communication needed.     │
 * │                                                                     │
 * │  Phase 2  User means    Each rank computes means for its rows,    │
 * │                          then MPI_Allgatherv assembles the full    │
 * │                          user_mean[] array on every rank.          │
 * │                                                                     │
 * │  Phase 3  Similarity    Each rank fills its rows of sim_matrix,   │
 * │                          then MPI_Allgatherv assembles the full    │
 * │                          N_USERS×N_USERS matrix on every rank.     │
 * │                                                                     │
 * │  Phase 4  Predictions   Each rank predicts for its users only     │
 * │                          (full sim_matrix is already available).   │
 * │                                                                     │
 * │  Phase 5  MAE           Each rank contributes partial error;      │
 * │                          MPI_Reduce sums them to rank 0.           │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * Compile:
 *   mpicc -O2 -Wall -o mpi_rec mpi/mpi_recommender.c -lm
 *
 * Run:
 *   mpirun -np 1  ./mpi_rec
 *   mpirun -np 2  ./mpi_rec
 *   mpirun -np 4  ./mpi_rec
 *   mpirun -np 4  ./mpi_rec 2000 1500
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>       /* MPI header – provides all MPI_* functions and types */

/* ── Fixed algorithm parameters (identical across all project versions) ─── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f   /* fraction of entries that are unrated         */
#define TOP_K            20    /* max neighbours used in the prediction formula */
#define SEED             42    /* RNG seed – must match serial for same dataset */
#define TEST_RATIO      0.10f  /* fraction of known ratings held out for MAE   */

/* ── Runtime sizes and MPI state (set in main) ──────────────────────────── */
static int N_USERS;
static int N_ITEMS;
static int mpi_rank;  /* this process's index within MPI_COMM_WORLD, 0-based  */
static int mpi_size;  /* total number of MPI processes launched               */

/*
 * Data arrays – allocated on EVERY rank.
 *
 * Each rank ultimately holds the full ratings[], user_mean[], and sim_matrix[]
 * arrays.  Only the predictions[] array is partially filled: each rank writes
 * only the rows it owns.  Allocating full arrays everywhere simplifies the
 * code at the cost of replicated memory (acceptable for this dataset scale).
 */
static float *ratings;      /* [N_USERS × N_ITEMS]  – 0.0 means "unrated"    */
static float *user_mean;    /* [N_USERS]             – mean rating per user   */
static float *sim_matrix;   /* [N_USERS × N_USERS]  – Pearson similarities   */
static float *predictions;  /* [N_USERS × N_ITEMS]  – predicted ratings      */

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* ── 2-D indexing macros ─────────────────────────────────────────────────── */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

/*
 * MPI_Wtime() is the MPI standard wall-clock timer.
 * It returns seconds as a double, is available on all ranks, and is
 * guaranteed to be monotonic.  We prefer it over clock_gettime() here
 * because it is defined by the MPI standard and is portable.
 */
static inline double now_sec(void) { return MPI_Wtime(); }

/* ─────────────────────────────────────────────────────────────────────────
 * get_range() – compute the [start, end) row slice owned by a given rank.
 *
 * Ceiling division ensures full coverage of N_USERS rows without gaps,
 * even when N_USERS is not evenly divisible by mpi_size.  The last rank
 * may receive fewer rows than earlier ranks but never more.
 *
 * This function is called both for the local rank and for every other rank
 * (when building the recvcounts/displs arrays for collective operations).
 * ───────────────────────────────────────────────────────────────────────── */
static void get_range(int rank, int size, int *start, int *end)
{
    int chunk = (N_USERS + size - 1) / size;  /* ceiling division */
    *start = rank * chunk;
    *end   = *start + chunk;
    if (*end > N_USERS) *end = N_USERS;
}

/* ── Memory allocation / deallocation ───────────────────────────────────── */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", mpi_rank);
        /*
         * MPI_Abort terminates all processes in the communicator immediately.
         * Use it instead of exit() when an unrecoverable error occurs inside
         * an MPI program, to prevent other ranks from hanging indefinitely
         * waiting for a collective operation that will never arrive.
         */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static void free_arrays(void)
{
    free(ratings);
    free(user_mean);
    free(sim_matrix);
    free(predictions);
    free(test_set);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 1 – Data generation  (every rank independently, no communication)
 *
 * All ranks call srand(SEED) and execute the exact same sequence of rand()
 * calls.  Because C's rand() is a deterministic algorithm seeded by a fixed
 * value, every rank produces an identical ratings matrix and test_set without
 * any inter-rank communication.
 *
 * This approach avoids the need to broadcast a large ratings array and avoids
 * defining a custom MPI datatype for TestEntry.  It is correct because
 * generate_data() is a pure function of SEED.
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
                R(u, i) = rating;
            }
        }
    }

    /* Only rank 0 prints to avoid N duplicate lines */
    if (mpi_rank == 0)
        printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
               "Test ratings: %d\n",
               N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 2 – User means  (local computation + MPI_Allgatherv)
 *
 * Step A: Each rank computes user_mean[u] for u in [start_u, end_u).
 *
 * Step B: MPI_Allgatherv assembles the full user_mean[] on every rank.
 *
 * ── MPI_Allgatherv explained ─────────────────────────────────────────────
 *
 *  "All"       → every rank both sends data AND receives the combined result.
 *  "gather"    → pieces from all ranks are collected into one large buffer.
 *  "v"         → variable counts: different ranks may contribute different
 *                numbers of elements (needed when N_USERS % mpi_size != 0).
 *
 *  sendbuf     → pointer to this rank's contribution in its local array.
 *  sendcount   → how many elements this rank contributes.
 *  recvbuf     → the full destination array (same on all ranks).
 *  recvcounts  → array[size]: how many elements each rank contributes.
 *  displs      → array[size]: offset in recvbuf where each rank's data goes.
 *
 *  After the call, recvbuf is identical on every rank:
 *    recvbuf[displs[r] .. displs[r]+recvcounts[r]) = rank r's sendbuf.
 *
 * Example (4 ranks, 10 users):
 *   Rank 0 owns users 0-2  (3 users) → recvcounts[0]=3, displs[0]=0
 *   Rank 1 owns users 3-5  (3 users) → recvcounts[1]=3, displs[1]=3
 *   Rank 2 owns users 6-7  (2 users) → recvcounts[2]=2, displs[2]=6
 *   Rank 3 owns users 8-9  (2 users) → recvcounts[3]=2, displs[3]=8
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_user_means(int start_u, int end_u,
                                int *recvcounts, int *displs)
{
    /* ── Step A: local computation for this rank's users ── */
    for (int u = start_u; u < end_u; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }

    /* ── Step B: share all slices so every rank has the complete array ── */
    MPI_Allgatherv(
        &user_mean[start_u],   /* sendbuf: start of this rank's slice         */
        end_u - start_u,       /* sendcount: number of floats this rank sends */
        MPI_FLOAT,             /* MPI type matching float                      */
        user_mean,             /* recvbuf: full array (already allocated)      */
        recvcounts,            /* how many floats each rank contributes        */
        displs,                /* where in recvbuf each rank's data is placed  */
        MPI_FLOAT,
        MPI_COMM_WORLD         /* communicator = all ranks in this job         */
    );
    /*
     * After this call, user_mean[] is fully populated on every rank.
     * This is required because pearson_similarity(u, v) reads user_mean[v]
     * for all v, not just the v values this rank is responsible for.
     */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3 – Pearson similarity  (local computation + MPI_Allgatherv)
 *
 * pearson_similarity() reads only ratings[] and user_mean[], both of which
 * are fully populated on every rank after Phase 1 and Phase 2 respectively.
 * It is stateless and safe to call concurrently.
 *
 * Each rank computes ALL columns for its owned rows (v = 0..N_USERS-1), not
 * just the upper triangle.  This means the same (u,v) pair may be computed
 * by two different ranks (the one that owns row u and the one that owns row v)
 * but the values are identical since Pearson correlation is symmetric.
 * The slight extra computation simplifies the MPI gather pattern considerably:
 * each rank sends exactly (local_rows × N_USERS) contiguous floats.
 * ═══════════════════════════════════════════════════════════════════════════ */
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

static void compute_all_similarities(int start_u, int end_u,
                                      int *recvcounts, int *displs)
{
    /* ── Step A: fill all columns of this rank's rows ── */
    for (int u = start_u; u < end_u; u++) {
        SIM(u, u) = 1.0f;                       /* self-similarity is always 1 */
        for (int v = 0; v < N_USERS; v++) {
            if (v == u) continue;
            SIM(u, v) = pearson_similarity(u, v);
        }
    }

    /*
     * ── Step B: assemble the full N_USERS×N_USERS matrix on every rank ──
     *
     * Each rank sends (end_u - start_u) * N_USERS floats starting at
     * &sim_matrix[start_u * N_USERS].  recvcounts[r] and displs[r] are
     * already scaled by N_USERS (set up in main).
     *
     * After this collective, sim_matrix[] is identical and complete on every
     * rank.  The prediction phase can then look up SIM(u, v) for any (u, v)
     * without further communication.
     */
    MPI_Allgatherv(
        &sim_matrix[(size_t)start_u * N_USERS], /* sendbuf: this rank's row block */
        (end_u - start_u) * N_USERS,    /* sendcount: row_count × N_USERS     */
        MPI_FLOAT,
        sim_matrix,                      /* recvbuf: full matrix               */
        recvcounts,                      /* floats contributed by each rank    */
        displs,                          /* byte-offset of each rank's block   */
        MPI_FLOAT,
        MPI_COMM_WORLD
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 4 – Predictions  (purely local, no communication)
 *
 * Every rank now holds the complete sim_matrix[] and ratings[].  Each rank
 * independently predicts ratings for its own users [start_u, end_u) using
 * the weighted-average formula with TOP_K most similar neighbours.
 *
 * No MPI communication is needed in this phase.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

static void compute_all_predictions(int start_u, int end_u)
{
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
    if (!nbrs) {
        fprintf(stderr, "Rank %d: malloc failed for nbrs.\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int u = start_u; u < end_u; u++) {
        for (int item = 0; item < N_ITEMS; item++) {

            /* Keep the actual rating for items the user already rated */
            if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

            /* Collect neighbours who rated this item with positive similarity */
            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u || R(v, item) == 0.0f) continue;
                float s = SIM(u, v);
                if (s <= 0.0f) continue;
                nbrs[cnt].idx = v;
                nbrs[cnt].val = s;
                cnt++;
            }

            if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

            /* Sort by similarity descending, keep best TOP_K */
            qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
            int k = (cnt < TOP_K) ? cnt : TOP_K;

            /* Weighted sum: user_mean + Σ(sim_j × deviation_j) / Σ sim_j */
            double num = 0.0, den = 0.0;
            for (int j = 0; j < k; j++) {
                float s = nbrs[j].val;
                num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                den += s;
            }

            float pred = (den > 1e-10)
                         ? user_mean[u] + (float)(num / den)
                         : user_mean[u];
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 5 – MAE evaluation  (partial sum + MPI_Reduce)
 *
 * All ranks generated the same test_set (same seed).  Each rank iterates
 * through the test_set but only accumulates error for entries whose user
 * falls in its owned range [start_u, end_u) — because only those predictions
 * were computed by this rank.
 *
 * ── MPI_Reduce explained ─────────────────────────────────────────────────
 *
 *  MPI_Reduce combines one value per rank into a single result on one rank
 *  (the "root"), using a specified operation (here MPI_SUM).
 *
 *  sendbuf  → address of this rank's local value.
 *  recvbuf  → address where the result is written (meaningful only on root).
 *  count    → number of elements to combine.
 *  op       → reduction operation: MPI_SUM, MPI_MAX, MPI_MIN, etc.
 *  root     → rank that receives the final result (rank 0 here).
 *
 *  After the call, global_err on rank 0 = sum of local_err across all ranks.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float evaluate_mae(int start_u, int end_u)
{
    double local_err = 0.0;
    int    local_cnt = 0;

    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        if (u < start_u || u >= end_u) continue;  /* not this rank's user */
        local_err += fabs(PRED(u, test_set[t].item) - test_set[t].rating);
        local_cnt++;
    }

    double global_err = 0.0;
    int    global_cnt = 0;

    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    /* Only rank 0 has valid global_err after MPI_Reduce */
    if (mpi_rank == 0 && global_cnt > 0)
        return (float)(global_err / global_cnt);
    return 0.0f;
}

/*
 * similarity_checksum – called only on rank 0 after MPI_Allgatherv.
 *
 * After the Allgatherv in Phase 3, every rank holds the complete sim_matrix.
 * Rank 0 can therefore sum the full matrix directly without another Reduce.
 * This value must match the serial/OpenMP/Pthreads checksums for the same
 * input size and seed, confirming numerical correctness.
 */
static double similarity_checksum(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /*
     * MPI_Init() MUST be the first MPI call in the program.
     * It initialises the communication infrastructure, assigns rank IDs,
     * and enables all subsequent MPI operations.
     * Passing &argc and &argv allows MPI to strip its own flags from the
     * argument list before the application parses its own arguments.
     */
    MPI_Init(&argc, &argv);

    /*
     * MPI_Comm_rank: "what is my rank within this communicator?"
     * MPI_Comm_size: "how many ranks are in this communicator?"
     *
     * MPI_COMM_WORLD is the default communicator that includes every rank
     * launched by mpirun.  Rank 0 is conventionally the "master" process
     * that handles I/O.
     */
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

    /* Compute this rank's user row range */
    int start_u, end_u;
    get_range(mpi_rank, mpi_size, &start_u, &end_u);

    if (mpi_rank == 0) {
        printf("=== Pearson Correlation Recommender – MPI Version ===\n");
        printf("    Users: %d | Items: %d | Top-K: %d | Processes: %d\n\n",
               N_USERS, N_ITEMS, TOP_K, mpi_size);
    }

    /*
     * Build recvcounts[] and displs[] arrays for MPI_Allgatherv calls.
     *
     * These arrays describe how the full receive buffer is partitioned:
     *   recvcounts[r] = number of elements rank r contributes.
     *   displs[r]     = index in the receive buffer where rank r's data starts.
     *
     * We pre-compute two variants:
     *   mean_*  → for user_mean[] (1 float per owned user row)
     *   sim_*   → for sim_matrix[] (N_USERS floats per owned user row)
     *
     * Building them once here avoids recomputing inside every function call.
     */
    int *mean_recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *mean_displs     = (int *)malloc(mpi_size * sizeof(int));
    int *sim_recvcounts  = (int *)malloc(mpi_size * sizeof(int));
    int *sim_displs      = (int *)malloc(mpi_size * sizeof(int));

    for (int r = 0; r < mpi_size; r++) {
        int s, e;
        get_range(r, mpi_size, &s, &e);
        int local_rows       = e - s;
        mean_recvcounts[r]   = local_rows;              /* 1 float per user   */
        mean_displs[r]       = s;
        sim_recvcounts[r]    = local_rows * N_USERS;    /* N_USERS floats/row */
        sim_displs[r]        = s * N_USERS;
    }

    alloc_arrays();

    double t0, t1, t_sim, t_pred;

    /* ── Phase 1: Data generation ──────────────────────────────────────── */
    /*
     * MPI_Barrier: a synchronisation fence.  No rank passes this call until
     * ALL ranks have reached it.  We use it before each timed section so that
     * the timer on rank 0 reflects the true wall-clock cost including any
     * load imbalance (if one rank is slower reaching the barrier, everyone
     * waits and the timer captures that slack).
     */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    generate_data();
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* ── Phase 2: User means ────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_user_means(start_u, end_u, mean_recvcounts, mean_displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] User mean compute  : %.4f s  [MPI, %d processes]\n",
               t1 - t0, mpi_size);

    /* ── Phase 3: Similarity matrix ─────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_all_similarities(start_u, end_u, sim_recvcounts, sim_displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t_sim = now_sec() - t0;
    if (mpi_rank == 0) {
        printf("[Timing] Similarity matrix  : %.4f s  [MPI, %d processes]\n",
               t_sim, mpi_size);
        /* Full sim_matrix is available on rank 0 after Allgatherv */
        printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());
    }

    /* ── Phase 4: Predictions ───────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_all_predictions(start_u, end_u);
    MPI_Barrier(MPI_COMM_WORLD);
    t_pred = now_sec() - t0;
    if (mpi_rank == 0)
        printf("[Timing] Prediction phase   : %.4f s  [MPI, %d processes]\n",
               t_pred, mpi_size);

    /* ── Phase 5: Evaluation (distributed Reduce) ──────────────────────── */
    float mae = evaluate_mae(start_u, end_u);
    if (mpi_rank == 0) {
        printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
               mae, test_size);
        printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);
    }

    /*
     * Sample predictions – printed only by rank 0.
     *
     * Rank 0 owns users [0, end_u).  For any practical run (N_USERS ≥ 10
     * and mpi_size ≤ 64), end_u > 5, so users 0-4 are always on rank 0.
     */
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

    /*
     * MPI_Finalize() MUST be the last MPI call.
     * It shuts down the MPI communication layer cleanly.
     * Any MPI call after this point is undefined behaviour.
     */
    MPI_Finalize();
    return 0;
}
