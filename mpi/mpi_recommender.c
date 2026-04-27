/* MPI Distributed Memory Pearson Correlation Recommender
 * 
 * Compile:
 *   mpicc -O2 -Wall -o mpi_rec mpi_recommender.c -lm
 * 
 * Run:
 *   mpirun -np 4 ./mpi_rec
 *   mpirun -np 8 ./mpi_rec 500 300
 *   mpirun -np 4 ./mpi_rec 2000 1500
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
#define SPARSITY       0.70f   /* fraction of entries left unrated            */
#define TOP_K            20    /* neighbours used in prediction               */
#define SEED             42    /* RNG seed for reproducibility                */
#define TEST_RATIO      0.10f  /* fraction of known ratings held out for MAE  */

/* ── MPI variables ───────────────────────────────────────────────────────── */
static int rank, num_procs;
static MPI_Comm comm = MPI_COMM_WORLD;

/* ── Runtime size variables (set in main) ────────────────────────────────── */
static int N_USERS;
static int N_ITEMS;

/* ── Local and global data structures ────────────────────────────────────── */
static float *ratings;          /* [N_USERS × N_ITEMS] – 0 = unrated         */
static float *user_mean;        /* [N_USERS]                                  */
static float *sim_matrix;       /* [N_USERS × N_USERS] (full, on each proc)  */
static float *predictions;      /* [N_USERS × N_ITEMS]                       */
static float *local_sim_rows;   /* [local_num_rows × N_USERS] (this proc)    */

static int local_user_start;    /* First user assigned to this rank           */
static int local_user_end;      /* Last user assigned to this rank            */
static int local_num_users;     /* Number of users assigned to this rank      */

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* ── Convenience macros ──────────────────────────────────────────────────── */
#define R(u,i)         ratings[(u)*N_ITEMS + (i)]
#define SIM(u,v)       sim_matrix[(u)*N_USERS + (v)]
#define PRED(u,i)      predictions[(u)*N_ITEMS + (i)]
#define LOCAL_SIM(r,v) local_sim_rows[(r)*N_USERS + (v)]

/* ── Timing ──────────────────────────────────────────────────────────────── */
static inline double now_sec(void)
{
    return MPI_Wtime();
}

/* ── Phase 1: Allocate memory ────────────────────────────────────────────── */
static void alloc_arrays(void)
{
    if (rank == 0) {
        ratings  = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
        user_mean = (float *)calloc(N_USERS,          sizeof(float));
        predictions = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
    }

    /* All processes get full sim_matrix for easy broadcasting */
    sim_matrix = (float *)calloc(N_USERS * N_USERS, sizeof(float));

    /* Local similarity rows for this process */
    local_sim_rows = (float *)calloc(local_num_users * N_USERS, sizeof(float));

    if (!sim_matrix || !local_sim_rows) {
        fprintf(stderr, "[Rank %d] Error: memory allocation failed.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == 0 && (!ratings || !user_mean || !predictions)) {
        fprintf(stderr, "[Rank 0] Error: memory allocation failed.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

static void free_arrays(void)
{
    if (rank == 0) {
        free(ratings);
        free(user_mean);
        free(predictions);
        free(test_set);
    }
    free(sim_matrix);
    free(local_sim_rows);
}

/* ── Phase 2: Data generation (serial on rank 0) ──────────────────────────── */
static void generate_data(void)
{
    if (rank != 0) return;

    srand(SEED);

    int capacity = (int)(N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
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

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ── Phase 3: Broadcast data from rank 0 ─────────────────────────────────── */
static void broadcast_data(void)
{
    MPI_Bcast(&test_size, 1, MPI_INT, 0, comm);

    if (rank != 0) {
        ratings  = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
        user_mean = (float *)calloc(N_USERS,         sizeof(float));
        predictions = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
        test_set = (TestEntry *)malloc((test_size + 100) * sizeof(TestEntry));
    }

    MPI_Bcast(ratings,  N_USERS * N_ITEMS, MPI_FLOAT, 0, comm);
    MPI_Bcast(user_mean, N_USERS,          MPI_FLOAT, 0, comm);
    MPI_Bcast(test_set, test_size, MPI_BYTE, 0, comm);
}

/* ── Phase 4: Compute user means (parallel) ──────────────────────────────── */
static void compute_user_means_distributed(void)
{
    float *local_means = (rank == 0) ? user_mean : (float *)calloc(N_USERS, sizeof(float));

    /* Compute means for all users on rank 0 (or distributed if needed) */
    if (rank == 0) {
        for (int u = 0; u < N_USERS; u++) {
            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < N_ITEMS; i++) {
                if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
            }
            user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
        }
    }

    /* Broadcast user means to all processes */
    MPI_Bcast(user_mean, N_USERS, MPI_FLOAT, 0, comm);

    if (rank != 0) free(local_means);
}

/* ── Phase 5: Pearson similarity – compute locally assigned rows ──────────── */
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

/* Each process computes its assigned rows of the similarity matrix */
static void compute_local_similarities(void)
{
    /* Each rank computes its assigned users' similarity to all users */
    for (int u_local = 0; u_local < local_num_users; u_local++) {
        int u = local_user_start + u_local;

        /* Diagonal is always 1.0 */
        LOCAL_SIM(u_local, u) = 1.0f;

        /* Compute similarity with all other users */
        for (int v = 0; v < N_USERS; v++) {
            if (u != v) {
                float s = pearson_similarity(u, v);
                LOCAL_SIM(u_local, v) = s;
            }
        }
    }
}

/* Gather all similarity rows using Allgatherv */
static void gather_similarities(void)
{
    /* Prepare data for Allgatherv */
    int *recvcounts = (int *)malloc(num_procs * sizeof(int));
    int *displs = (int *)malloc(num_procs * sizeof(int));

    for (int p = 0; p < num_procs; p++) {
        int users_per_proc = N_USERS / num_procs;
        int extra = N_USERS % num_procs;
        int start = p * users_per_proc + (p < extra ? p : extra);
        int count = users_per_proc + (p < extra ? 1 : 0);
        
        recvcounts[p] = count * N_USERS;
        displs[p] = start * N_USERS;
    }

    MPI_Allgatherv(
        local_sim_rows, local_num_users * N_USERS, MPI_FLOAT,
        sim_matrix, recvcounts, displs, MPI_FLOAT,
        comm
    );

    free(recvcounts);
    free(displs);
}

/* ── Phase 6: Predictions – compute locally assigned users ────────────────── */
typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

/* Each process computes predictions for its assigned users */
static void compute_local_predictions(void)
{
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));

    for (int u_local = 0; u_local < local_num_users; u_local++) {
        int u = local_user_start + u_local;

        for (int item = 0; item < N_ITEMS; item++) {
            if (R(u, item) != 0.0f) {
                PRED(u, item) = R(u, item);
                continue;
            }

            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u || R(v, item) == 0.0f) continue;
                float s = SIM(u, v);
                if (s <= 0.0f) continue;
                nbrs[cnt].idx = v;
                nbrs[cnt].val = s;
                cnt++;
            }

            if (cnt == 0) {
                PRED(u, item) = user_mean[u];
                continue;
            }

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
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
}

/* Gather all predictions using Allgatherv */
static void gather_predictions(void)
{
    /* Prepare data for Allgatherv */
    int *recvcounts = (int *)malloc(num_procs * sizeof(int));
    int *displs = (int *)malloc(num_procs * sizeof(int));

    for (int p = 0; p < num_procs; p++) {
        int users_per_proc = N_USERS / num_procs;
        int extra = N_USERS % num_procs;
        int start = p * users_per_proc + (p < extra ? p : extra);
        int count = users_per_proc + (p < extra ? 1 : 0);
        
        recvcounts[p] = count * N_ITEMS;
        displs[p] = start * N_ITEMS;
    }

    float *local_preds = (float *)malloc(local_num_users * N_ITEMS * sizeof(float));
    for (int u = 0; u < local_num_users; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            local_preds[u * N_ITEMS + i] = PRED(local_user_start + u, i);
        }
    }

    MPI_Allgatherv(
        local_preds, local_num_users * N_ITEMS, MPI_FLOAT,
        predictions, recvcounts, displs, MPI_FLOAT,
        comm
    );

    free(local_preds);
    free(recvcounts);
    free(displs);
}

/* ── Phase 7: Evaluation ─────────────────────────────────────────────────── */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;

    double local_err = 0.0;
    for (int t = 0; t < test_size; t++) {
        local_err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    }

    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) return (float)(global_err / test_size);
    return 0.0f;
}

static double similarity_checksum(void)
{
    double local_sum = 0.0;
    for (int u = local_user_start; u < local_user_end; u++) {
        for (int v = 0; v < N_USERS; v++) {
            local_sum += SIM(u, v);
        }
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    return global_sum;
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Distribute users evenly across processes */
    int users_per_proc = N_USERS / num_procs;
    int extra_users = N_USERS % num_procs;

    local_user_start = rank * users_per_proc + (rank < extra_users ? rank : extra_users);
    local_num_users = users_per_proc + (rank < extra_users ? 1 : 0);
    local_user_end = local_user_start + local_num_users;

    if (rank == 0) {
        printf("=== Pearson Correlation Recommender – MPI Version ===\n");
        printf("    Users: %d | Items: %d | Top-K: %d | MPI Processes: %d\n\n",
               N_USERS, N_ITEMS, TOP_K, num_procs);
    }

    double t0, t1, t_sim, t_pred, t_total;

    /* Synchronize start */
    MPI_Barrier(comm);
    t_total = now_sec();

    alloc_arrays();

    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    t0 = now_sec();
    broadcast_data();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Data broadcast     : %.4f s\n", t1 - t0);

    t0 = now_sec();
    compute_user_means_distributed();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] User mean compute  : %.4f s  [%d procs]\n", t1 - t0, num_procs);

    t0 = now_sec();
    compute_local_similarities();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Local sim compute  : %.4f s  [%d procs]\n", t1 - t0, num_procs);

    t0 = now_sec();
    gather_similarities();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Similarity gather  : %.4f s\n", t1 - t0);

    t_sim = t1 - t0;  // Last time measurement
    for (int i = 0; i < num_procs; i++) {
        double local_t = (i == 0) ? t_sim : 0.0;
        MPI_Bcast(&t_sim, 1, MPI_DOUBLE, 0, comm);
    }
    if (rank == 0) printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    t0 = now_sec();
    compute_local_predictions();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Local pred compute : %.4f s  [%d procs]\n", t1 - t0, num_procs);

    t0 = now_sec();
    gather_predictions();
    t1 = now_sec();
    if (rank == 0) printf("[Timing] Predictions gather : %.4f s\n", t1 - t0);

    t_pred = t1 - t0;
    if (rank == 0) {
        printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
               evaluate_mae(), test_size);
    }

    MPI_Barrier(comm);
    t1 = now_sec();

    if (rank == 0) {
        printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);
        printf("[Timing] Total wall-clock   : %.4f s\n", t1 - t_total);

        /* Sample output */
        int show_u = (N_USERS < 5) ? N_USERS : 5;
        int show_i = (N_ITEMS < 5) ? N_ITEMS : 5;
        printf("\n--- Sample Predictions (first %d users, %d items) ---\n", show_u, show_i);
        printf("%-9s", "User\\Item");
        for (int i = 0; i < show_i; i++) printf("  Item%-3d", i);
        printf("\n");
        for (int u = 0; u < show_u; u++) {
            printf("User %-4d", u);
            for (int i = 0; i < show_i; i++) printf("  %5.2f  ", PRED(u, i));
            printf("\n");
        }
    }

    free_arrays();
    MPI_Finalize();
    return 0;
}
