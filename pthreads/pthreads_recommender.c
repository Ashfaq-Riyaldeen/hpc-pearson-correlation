#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

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

typedef struct {
    int tid;
    int nthreads;
} ThreadArgs;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void work_range(int total, int tid, int nthreads, int *start, int *end)
{
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

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

static void *thread_user_means(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int start, end;
    work_range(N_USERS, a->tid, a->nthreads, &start, &end);

    for (int u = start; u < end; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
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

static void *thread_similarities(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int start, end;
    work_range(N_USERS, a->tid, a->nthreads, &start, &end);

    for (int u = start; u < end; u++) {
        SIM(u, u) = 1.0f;
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;
            SIM(v, u) = s;
        }
    }
    return NULL;
}

typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

static void *thread_predictions(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int start, end;
    work_range(N_USERS, a->tid, a->nthreads, &start, &end);

    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
    if (!nbrs) {
        fprintf(stderr, "Thread %d: malloc failed.\n", a->tid);
        return NULL;
    }

    for (int u = start; u < end; u++) {
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
    return NULL;
}

static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    return (float)(err / test_size);
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
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
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
        if (t > 0) fputc(',', fp);
        fprintf(fp, "%.17g",
                (double)PRED(test_set[t].user, test_set[t].item));
    }
    fputs("]\n", fp);
    fclose(fp);
    printf("[JSON]   Test predictions written to %s (%d values)\n",
           path, test_size);
}

static double run_parallel(void *(*fn)(void *),
                            pthread_t  *threads,
                            ThreadArgs *args)
{
    double t0 = now_sec();

    for (int t = 0; t < N_THREADS; t++) {
        args[t].tid      = t;
        args[t].nthreads = N_THREADS;
        if (pthread_create(&threads[t], NULL, fn, &args[t]) != 0) {
            fprintf(stderr, "Error: pthread_create failed for thread %d\n", t);
            exit(EXIT_FAILURE);
        }
    }

    for (int t = 0; t < N_THREADS; t++)
        pthread_join(threads[t], NULL);

    return now_sec() - t0;
}

int main(int argc, char *argv[])
{
    N_USERS   = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS   = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;
    N_THREADS = (argc >= 4) ? atoi(argv[3]) : DEFAULT_THREADS;

    if (N_USERS <= 0 || N_ITEMS <= 0 || N_THREADS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items] [num_threads]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (N_THREADS > MAX_THREADS) {
        fprintf(stderr, "Warning: capping threads at %d\n", MAX_THREADS);
        N_THREADS = MAX_THREADS;
    }

    printf("=== Pearson Correlation Recommender – Pthreads Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d | Threads: %d\n\n",
           N_USERS, N_ITEMS, TOP_K, N_THREADS);

    pthread_t  threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];

    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    t0 = now_sec();
    run_parallel(thread_user_means, threads, args);
    t1 = now_sec();
    printf("[Timing] User mean compute  : %.4f s  [pthreads, %d threads]\n",
           t1 - t0, N_THREADS);

    t_sim = run_parallel(thread_similarities, threads, args);
    printf("[Timing] Similarity matrix  : %.4f s  [pthreads, %d threads]\n",
           t_sim, N_THREADS);
    printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    t_pred = run_parallel(thread_predictions, threads, args);
    printf("[Timing] Prediction phase   : %.4f s  [pthreads, %d threads]\n",
           t_pred, N_THREADS);

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
