/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* Constants */
#define DEFAULT_USERS    1000
#define DEFAULT_ITEMS    1000
#define DEFAULT_THREADS     4
#define MAX_THREADS        64
#define SPARSITY         0.70f
#define TOP_K              20
#define SEED               42
#define TEST_RATIO        0.10f

/* Runtime sizes */
static int N_USERS;
static int N_ITEMS;
static int N_THREADS;

/* Global arrays */
static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

/* Test set */
typedef struct {
    int user;
    int item;
    float rating;
} TestEntry;

static TestEntry *test_set;
static int test_size;

/* Macros */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

/* Thread arguments */
typedef struct {
    int tid;
    int nthreads;
} ThreadArgs;

/* Timer */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Work distribution */
static void work_range(int total, int tid, int nthreads, int *start, int *end)
{
    int chunk = (total + nthreads - 1) / nthreads;

    *start = tid * chunk;
    *end = *start + chunk;

    if (*end > total)
        *end = total;
}

/* Memory allocation */
static void alloc_arrays(void)
{
    ratings = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean = (float *)calloc(N_USERS, sizeof(float));
    sim_matrix = (float *)calloc((size_t)N_USERS * N_USERS, sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        fprintf(stderr, "Memory allocation failed\n");
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

/* Data generation */
static void generate_data(void)
{
    srand(SEED);

    int capacity =
        (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;

    test_set = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {

            if ((float)rand() / RAND_MAX < SPARSITY)
                continue;

            float rating = (float)(rand() % 5) + 1.0f;

            if ((float)rand() / RAND_MAX < TEST_RATIO &&
                test_size < capacity) {

                test_set[test_size].user = u;
                test_set[test_size].item = i;
                test_set[test_size].rating = rating;
                test_size++;

            } else {
                R(u, i) = rating;
            }
        }
    }

    printf("[Data] Users: %d | Items: %d | Test ratings: %d\n",
           N_USERS, N_ITEMS, test_size);
}