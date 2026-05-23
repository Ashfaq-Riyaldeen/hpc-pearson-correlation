#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
 * Serial user-based collaborative filtering recommender.
 *
 * The program creates a synthetic sparse user-item ratings matrix, hides a
 * small fraction of known ratings as a test set, and predicts missing ratings
 * with Pearson-correlation neighbors.  It is the baseline implementation used
 * for comparing the parallel OpenMP, Pthreads, MPI, hybrid, and CUDA versions.
 *
 * High-level flow:
 *   1. Generate sparse ratings and hold out TEST_RATIO for evaluation.
 *   2. Compute each user's mean rating.
 *   3. Build the full user-user Pearson similarity matrix.
 *   4. Predict unrated items from the TOP_K most similar positive neighbors.
 *   5. Report MAE/RMSE and timing information.
 */

#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

static int N_USERS;
static int N_ITEMS;

static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* Store matrices as one-dimensional row-major arrays for compact allocation
 * and cache-friendly sequential access.  A zero rating means "unknown". */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

static void generate_data(void)
{
    srand(SEED);

    /* Approximate capacity for the held-out test ratings.  The exact number
     * depends on random sparsity and TEST_RATIO, so a small margin is added. */
    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            /* Skip most cells to simulate a sparse real-world rating matrix. */
            if ((float)rand() / RAND_MAX < SPARSITY) continue;
            float rating = (float)(rand() % 5) + 1.0f;

            /* Held-out ratings are not placed in ratings[]; they are used only
             * after prediction to measure how close the estimates were. */
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

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

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
        /* Use the middle of the 1..5 rating scale when a user has no observed
         * training ratings, which avoids division by zero and gives a neutral
         * baseline for predictions. */
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
}

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
            /* Pearson correlation compares how two users deviate from their
             * own average ratings on items both users have rated. */
            double du = R(u, i) - mu;
            double dv = R(v, i) - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    /* A single common item is too weak for a reliable correlation. */
    if (co < 2) return 0.0f;

    double denom = sqrt(den_u) * sqrt(den_v);
    /* If either user has no variation around their mean, correlation is
     * undefined; treat that pair as unrelated. */
    if (denom < 1e-10) return 0.0f;

    float s = (float)(num / denom);
    /* Clamp tiny floating-point overshoots back into Pearson's valid range. */
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return s;
}

static void compute_all_similarities(void)
{
    /* The matrix is symmetric: similarity(u,v) == similarity(v,u), so compute
     * only the upper triangle and mirror it into the lower triangle. */
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

typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    /* Sort from highest similarity to lowest similarity for Top-K selection. */
    return (fb > fa) - (fb < fa);
}

static void compute_all_predictions(void)
{
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));

    for (int u = 0; u < N_USERS; u++) {
        for (int item = 0; item < N_ITEMS; item++) {

            /* Known training ratings are copied directly; only missing entries
             * need collaborative-filtering estimates. */
            if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u) continue;
                if (R(v, item) == 0.0f) continue;
                /* This version uses only positive neighbors.  Negative
                 * correlations are ignored instead of being used inversely. */
                float s = SIM(u, v);
                if (s <= 0.0f) continue;
                nbrs[cnt].idx = v;
                nbrs[cnt].val = s;
                cnt++;
            }

            if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

            qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
            int k = (cnt < TOP_K) ? cnt : TOP_K;

            /* Standard mean-centered prediction:
             *   pred(u,i) = mean(u) + weighted average of
             *               neighbor_rating(v,i) - mean(v)
             * Similar neighbors therefore contribute their preference above or
             * below their normal rating behavior, not just their raw score. */
            double num = 0.0, den = 0.0;
            for (int j = 0; j < k; j++) {
                float s = nbrs[j].val;
                num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                den += s;
            }

            float pred = (den > 1e-10)
                         ? user_mean[u] + (float)(num / den)
                         : user_mean[u];

            /* Keep predictions within the synthetic rating scale. */
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
}

static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    return (float)(err / test_size);
}

static void dump_test_predictions_json(const char *path)
{
    /* Optional output used by benchmark/equivalence scripts.  Set
     * PRED_DUMP_PATH to write predictions for the held-out test entries. */
    if (!path) return;
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

static double similarity_checksum(void)
{
    /* Simple reproducibility check: parallel implementations can compare this
     * value with the serial result to catch major similarity-matrix mistakes. */
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

int main(int argc, char *argv[])
{
    /* Optional command-line arguments let benchmarks scale the problem size:
     *   ./serial_recommender [num_users] [num_items]
     */
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
        for (int i = 0; i < show_i; i++) printf("  %5.2f  ", PRED(u, i));
        printf("\n");
    }

    free_arrays();
    return 0;
}
