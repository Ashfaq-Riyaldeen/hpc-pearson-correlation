/* ============================================================================
 *  Pearson Correlation Recommender – Serial (single-threaded) baseline.
 *
 *  Pipeline:
 *    1. Generate a synthetic sparse user/item rating matrix.
 *    2. Compute each user's mean rating (used to center the data).
 *    3. Build the user-user similarity matrix via Pearson correlation.
 *    4. For every missing rating, predict using a Top-K weighted average of
 *       the most similar users who actually rated that item.
 *    5. Evaluate accuracy (MAE/RMSE) on a held-out test set.
 *
 *  This file is intentionally the simple reference implementation; the MPI +
 *  OpenMP hybrid version mirrors the same algorithm but parallelizes steps
 *  2-4 across processes and threads.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Problem-size and algorithm tuning constants.
 * SPARSITY     = fraction of entries left empty (0.70 -> ~30% filled).
 * TOP_K        = number of nearest neighbors used when predicting.
 * SEED         = fixed RNG seed so runs are reproducible/comparable.
 * TEST_RATIO   = fraction of generated ratings withheld for evaluation. */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

static int N_USERS;
static int N_ITEMS;

/* Global flat arrays (1-D buffers indexed via the macros below).
 *   ratings     : N_USERS x N_ITEMS rating matrix (0.0f means "no rating").
 *   user_mean   : average rating per user.
 *   sim_matrix  : N_USERS x N_USERS Pearson similarity matrix.
 *   predictions : N_USERS x N_ITEMS predicted ratings. */
static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

/* Held-out ratings used for MAE/RMSE evaluation (not visible to the model). */
typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* Row-major access helpers for the flat 2-D buffers above. */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

/* Monotonic wall-clock timer in seconds (immune to system time changes). */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Allocate all big buffers up front. calloc() zeros them, which matters: a
 * 0.0f entry in `ratings` is the sentinel meaning "user has not rated this
 * item", and `sim_matrix`/`predictions` start clean. */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

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

/* Build a synthetic sparse rating matrix.
 *
 * For each (user, item) cell we either:
 *   - skip it (probability = SPARSITY), leaving the cell as 0.0f, OR
 *   - draw a rating in {1,2,3,4,5} and EITHER place it into the training
 *     matrix OR (with probability TEST_RATIO) move it into the test set
 *     so the prediction step has nothing to "cheat" with at that cell. */
static void generate_data(void)
{
    srand(SEED);

    /* Upper bound on test-set size: at most all non-sparse cells. */
    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            if ((float)rand() / RAND_MAX < SPARSITY) continue;
            float rating = (float)(rand() % 5) + 1.0f;
            if ((float)rand() / RAND_MAX < TEST_RATIO && test_size < capacity) {
                /* Withhold this rating: model must predict it later. */
                test_set[test_size].user   = u;
                test_set[test_size].item   = i;
                test_set[test_size].rating = rating;
                test_size++;
            } else {
                R(u, i) = rating;
            }
        }
    }

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* Mean rating per user, ignoring empty cells. Users with no ratings get a
 * neutral 3.0 fallback so later subtractions are well-defined. */
static void compute_user_means(void)
{
    for (int u = 0; u < N_USERS; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) {
                sum += R(u, i);
                cnt++;
            }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
}

/* Pearson correlation between two users, computed only over items they have
 * BOTH rated (co-rated items). Returns a value in [-1, +1]:
 *   +1 -> tastes move together,   0 -> uncorrelated,   -1 -> opposite.
 *
 * Formula:        sum((Ru - mu) * (Rv - mv))
 *           -------------------------------------
 *           sqrt(sum((Ru-mu)^2)) * sqrt(sum((Rv-mv)^2))
 *
 * Edge cases handled:
 *   - fewer than 2 co-rated items  -> not enough signal, return 0.
 *   - denominator near zero        -> one user is constant on overlap, return 0.
 *   - tiny FP drift outside [-1,1] -> clamp. */
static float pearson_similarity(int u, int v)
{
    double num = 0.0;
    double den_u = 0.0;
    double den_v = 0.0;
    int    co  = 0;
    float  mu  = user_mean[u];
    float  mv  = user_mean[v];

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

/* Fill the full N x N similarity matrix. The matrix is symmetric, so we only
 * compute the upper triangle and mirror it into the lower half — this halves
 * the work compared to computing every (u,v) pair. */
static void compute_all_similarities(void)
{
    for (int u = 0; u < N_USERS; u++)
        SIM(u, u) = 1.0f;

    for (int u = 0; u < N_USERS; u++) {
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;
            SIM(v, u) = s;
        }
    }
}

/* (neighbor index, similarity score) pair, used for ranking candidates. */
typedef struct { int idx; float val; } SimPair;

/* qsort comparator: sort SimPairs by `val` in descending order so the most
 * similar neighbors land at the front of the array. */
static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

/* Predict every (user, item) cell using user-based collaborative filtering.
 *
 * For each (u, item):
 *   - If u already rated `item`, keep the actual rating.
 *   - Otherwise collect every OTHER user v who DID rate `item` and has a
 *     positive similarity to u (negative similarities are excluded — they
 *     anti-correlate and tend to hurt accuracy more than they help).
 *   - Sort those candidates by similarity, keep the Top-K best neighbors.
 *   - Predict by adding u's mean to the similarity-weighted average of how
 *     much those neighbors deviated from THEIR own mean on this item:
 *
 *         pred = mean(u) + ( sum_j sim(u,v_j) * (R(v_j,item) - mean(v_j)) )
 *                          ------------------------------------------------
 *                                       sum_j sim(u,v_j)
 *
 *   - Clamp to the [1, 5] rating scale. */
static void compute_all_predictions(void)
{
    /* Scratch buffer reused across all (u, item) pairs — at most N_USERS
     * candidate neighbors per cell, so size it once and avoid mallocing
     * inside the hot loop. */
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));

    for (int u = 0; u < N_USERS; u++) {
        for (int item = 0; item < N_ITEMS; item++) {

            /* Cell is observed: just copy it across, nothing to predict. */
            if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

            /* Gather candidate neighbors: rated this item, positively similar. */
            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u) continue;
                if (R(v, item) == 0.0f) continue;
                float s = SIM(u, v);
                if (s <= 0.0f) continue;
                nbrs[cnt].idx = v;
                nbrs[cnt].val = s;
                cnt++;
            }

            /* No usable neighbors -> fall back to u's average. */
            if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

            /* Pick the K most similar neighbors. */
            qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
            int k = (cnt < TOP_K) ? cnt : TOP_K;

            /* Similarity-weighted mean of mean-centered neighbor ratings. */
            double num = 0.0, den = 0.0;
            for (int j = 0; j < k; j++) {
                float s = nbrs[j].val;
                num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                den += s;
            }

            float pred = (den > 1e-10)
                         ? user_mean[u] + (float)(num / den)
                         : user_mean[u];

            /* Keep predictions on the valid 1..5 rating scale. */
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
}

/* Mean Absolute Error on the held-out test set:  mean( |pred - actual| ).
 * Lower is better; intuitive units (same scale as the ratings themselves). */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    return (float)(err / test_size);
}

/* Root Mean Squared Error: sqrt(mean((pred-actual)^2)).
 * Penalizes large mistakes more harshly than MAE. */
static float evaluate_rmse(void)
{
    if (test_size == 0) return 0.0f;
    double sq = 0.0;
    for (int t = 0; t < test_size; t++) {
        double d = PRED(test_set[t].user, test_set[t].item) - test_set[t].rating;
        sq += d * d;
    }
    return (float)sqrt(sq / test_size);
}

/* Cheap sanity check: summing the similarity matrix gives a single scalar we
 * can compare across runs and across versions (serial vs. hybrid) to confirm
 * the parallel implementation produced numerically equivalent results. */
static double similarity_checksum(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

/* Entry point: parse args, run each pipeline stage with timing, print a
 * small sample of predictions, and report accuracy. */
int main(int argc, char *argv[])
{
    /* Optional CLI overrides: ./serial_recommender [num_users] [num_items]. */
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("=== Pearson Correlation Recommender – Serial Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d\n\n",
           N_USERS, N_ITEMS, TOP_K);

    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    t0 = now_sec();
    compute_user_means();
    t1 = now_sec();
    printf("[Timing] User mean compute  : %.4f s\n", t1 - t0);

    t0 = now_sec();
    compute_all_similarities();
    t1 = now_sec();
    t_sim = t1 - t0;
    printf("[Timing] Similarity matrix  : %.4f s\n", t_sim);
    printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    t0 = now_sec();
    compute_all_predictions();
    t1 = now_sec();
    t_pred = t1 - t0;
    printf("[Timing] Prediction phase   : %.4f s\n", t_pred);

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
        for (int i = 0; i < show_i; i++) printf("  %5.2f  ", PRED(u, i));
        printf("\n");
    }

    free_arrays();
    return 0;
}
