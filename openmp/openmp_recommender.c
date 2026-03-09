 /* Compile:
 *   gcc -O2 -fopenmp -o openmp_rec openmp_recommender.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ── Fixed parameters ────────────────────────────────────────────────────── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f   /* fraction of entries left unrated            */
#define TOP_K            20    /* neighbours used in prediction               */
#define SEED             42    /* RNG seed for reproducibility                */
#define TEST_RATIO      0.10f  /* fraction of known ratings held out for MAE  */

/* ── Runtime size variables (set in main) ────────────────────────────────── */
static int N_USERS;
static int N_ITEMS;

/*
 * All matrices stored as flat 1-D arrays (dynamic allocation).
 * Access: ratings[u * N_ITEMS + i]
 */
static float *ratings;      /* [N_USERS × N_ITEMS] – 0 = unrated             */
static float *user_mean;    /* [N_USERS]                                      */
static float *sim_matrix;   /* [N_USERS × N_USERS]                           */
static float *predictions;  /* [N_USERS × N_ITEMS]                           */

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* ── Convenience macros ──────────────────────────────────────────────────── */
#define R(u,i)    ratings[(u)*N_ITEMS + (i)]
#define SIM(u,v)  sim_matrix[(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(u)*N_ITEMS + (i)]

/* ── Timing ──────────────────────────────────────────────────────────────── */
static inline double now_sec(void) { return omp_get_wtime(); }

/* ── Phase 1: Allocate memory ────────────────────────────────────────────── */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,            sizeof(float));
    sim_matrix  = (float *)calloc(N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc(N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        exit(EXIT_FAILURE);
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

/* ── Phase 2: Data generation (serial – identical seed to baseline) ──────── */
static void generate_data(void)
{
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
                /* leave R(u,i) = 0 → treated as unrated */
            } else {
                R(u, i) = rating;
            }
        }
    }

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ── Phase 3: User means – PARALLEL ─────────────────────────────────────── */
static void compute_user_means(void)
{
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < N_USERS; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
}

/* ── Phase 4: Similarity matrix – PARALLEL ───────────────────────────────── */
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

static void compute_all_similarities(void)
{
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < N_USERS; u++)
        SIM(u, u) = 1.0f;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int u = 0; u < N_USERS; u++) {
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;
            SIM(v, u) = s;
        }
    }
}

/* ── Phase 5: Predictions – PARALLEL ────────────────────────────────────── */
typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

static void compute_all_predictions(void)
{
    #pragma omp parallel
    {
        SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));

        #pragma omp for schedule(dynamic, 2)
        for (int u = 0; u < N_USERS; u++) {
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

                if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

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
}

/* ── Phase 6: Evaluation ─────────────────────────────────────────────────── */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    #pragma omp parallel for reduction(+:err) schedule(static)
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    return (float)(err / test_size);
}

static double similarity_checksum(void)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    /* Parse optional CLI args: ./openmp_rec [num_users] [num_items] */
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int nthreads;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    printf("=== Pearson Correlation Recommender – OpenMP Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d | Threads: %d\n\n",
           N_USERS, N_ITEMS, TOP_K, nthreads);

    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    t0 = now_sec(); generate_data();          t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    t0 = now_sec(); compute_user_means();     t1 = now_sec();
    printf("[Timing] User mean compute  : %.4f s  [parallel, %d threads]\n", t1-t0, nthreads);

    t0 = now_sec(); compute_all_similarities(); t1 = now_sec();
    t_sim = t1 - t0;
    printf("[Timing] Similarity matrix  : %.4f s  [parallel, %d threads]\n", t_sim, nthreads);
    printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    t0 = now_sec(); compute_all_predictions(); t1 = now_sec();
    t_pred = t1 - t0;
    printf("[Timing] Prediction phase   : %.4f s  [parallel, %d threads]\n", t_pred, nthreads);

    printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
           evaluate_mae(), test_size);
    printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);

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

    free_arrays();
    return 0;
}
