
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>

// -------- Experiment configuration --------

#define NODE_COUNT (1024 * 1024)
#define STEPS 100000
#define TRIALS 15
#define WARMUP_STEPS 10000

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

/*
    ROB Reverse Engineering - Experiment 2

    Processor: Intel Xeon Gold 5420+ (Sapphire Rapids)

    Experiment 1 findings:
    - Timing remained around 16-17 ticks through 96 NOPs.
    - Timing increased beginning around 128 NOPs.
    - This may reflect instruction throughput rather
      than actual ROB capacity.

    Experiment 2 improvements:
    1. Use two independent randomized pointer chains.
    2. Sweep a larger range of NOP counts.
    3. Run a NOP-only control experiment.
    4. Increase measurement iterations and trials.
    5. Compare timing of ROB and control experiments.

    Goal:
    Investigate whether a reproducible timing transition
    can be attributed to the ROB rather than instruction
    throughput or memory hierarchy effects.

    Precise exception verification will be performed
    in a separate experiment.
*/

// -------- Timing primitives --------

static inline uint64_t x86_tsc_start(void) {

    _mm_lfence();

    uint64_t t = __rdtsc();

    _mm_lfence();

    return t;
}

static inline uint64_t x86_tsc_stop(void) {

    unsigned aux;

    uint64_t t = __rdtscp(&aux);

    _mm_lfence();

    return t;
}

// -------- Pointer-chasing structure --------

typedef struct node {

    struct node *next;

    // Make each node occupy a 64-byte cache line
    unsigned char padding[64 - sizeof(void *)];

} Struct1;

_Static_assert(sizeof(Struct1) == 64,
               "Node size must be 64 bytes");

// Prevent compiler from eliminating final pointer results
static Struct1 * volatile sink_a;
static Struct1 * volatile sink_b;

// -------- Random number generator --------

static uint32_t xorshift32(uint32_t *state) {

    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return *state = x;
}

// -------- Randomized pointer-chasing cycle --------

static void make_random_cycle(
    Struct1 *nodes,
    size_t n,
    uint32_t seed
) {

    if (n == 0 ||
        n > SIZE_MAX / sizeof(size_t) ||
        n > UINT32_MAX) {

        fprintf(stderr, "Invalid node count\n");
        exit(EXIT_FAILURE);
    }

    size_t *order = malloc(n * sizeof(*order));

    if (!order) {

        fprintf(stderr, "Allocation failed\n");
        exit(EXIT_FAILURE);
    }

    uint32_t state = seed ? seed : 1u;

    // Initialize indices
    for (size_t i = 0; i < n; i++) {

        order[i] = i;
    }

    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {

        size_t j =
            (size_t)(xorshift32(&state) %
                     (uint32_t)(i + 1));

        size_t tmp = order[i];

        order[i] = order[j];
        order[j] = tmp;
    }

    // Connect the shuffled nodes into a cycle
    for (size_t i = 0; i < n; i++) {

        nodes[order[i]].next =
            &nodes[order[(i + 1) % n]];
    }

    free(order);
}

// -------- Baseline pointer-chasing function --------

static double chase_batch(
    Struct1 *p,
    uint64_t steps
) {

    uint64_t t0 = x86_tsc_start();

    for (uint64_t i = 0; i < steps; i++) {

        p = p->next;
    }

    uint64_t t1 = x86_tsc_stop();

    sink_a = p;

    return (double)(t1 - t0) / steps;
}

// -------- ROB experiment --------

// Each iteration:
// 1. Load from pointer chain A.
// 2. Execute N NOP instructions.
// 3. Load from independent pointer chain B.

#define DEFINE_ROB_TEST(N)                                      \
static __attribute__((noinline))                                \
double rob_test_##N(                                            \
    Struct1 *a, Struct1 *b, uint64_t steps) {                   \
                                                                 \
    uint64_t t0 = x86_tsc_start();                              \
                                                                 \
    for (uint64_t i = 0; i < steps; i++) {                       \
                                                                 \
        __asm__ volatile (                                      \
            "mov (%0), %0\n\t"                                  \
            ".rept " STRINGIFY(N) "\n\t"                         \
            "nop\n\t"                                           \
            ".endr\n\t"                                         \
            "mov (%1), %1\n\t"                                  \
            : "+r"(a), "+r"(b)                                   \
            :                                                    \
            : "memory"                                          \
        );                                                       \
    }                                                            \
                                                                 \
    uint64_t t1 = x86_tsc_stop();                               \
                                                                 \
    sink_a = a;                                                  \
    sink_b = b;                                                  \
                                                                 \
    return (double)(t1 - t0) / steps;                            \
}

// -------- NOP-only control experiment --------

// Measures the overhead of N NOP instructions
// without the two pointer-chasing loads.

#define DEFINE_NOP_CONTROL(N)                                   \
static __attribute__((noinline))                                \
double nop_control_##N(uint64_t steps) {                        \
                                                                 \
    uint64_t t0 = x86_tsc_start();                              \
                                                                 \
    for (uint64_t i = 0; i < steps; i++) {                       \
                                                                 \
        __asm__ volatile (                                      \
            ".rept " STRINGIFY(N) "\n\t"                         \
            "nop\n\t"                                           \
            ".endr\n\t"                                         \
            :                                                    \
            :                                                    \
            : "memory"                                          \
        );                                                       \
    }                                                            \
                                                                 \
    uint64_t t1 = x86_tsc_stop();                               \
                                                                 \
    return (double)(t1 - t0) / steps;                            \
}

// -------- Generate benchmark variants --------

// Use one list to generate both types of tests.
// This prevents mismatches between test tables.

#define TEST_COUNTS(X) \
    X(0)              \
    X(16)             \
    X(32)             \
    X(64)             \
    X(96)             \
    X(112)            \
    X(128)            \
    X(144)            \
    X(160)            \
    X(192)            \
    X(224)            \
    X(256)            \
    X(320)            \
    X(384)            \
    X(448)            \
    X(480)            \
    X(496)            \
    X(512)            \
    X(528)            \
    X(544)            \
    X(576)            \
    X(640)            \
    X(768)

TEST_COUNTS(DEFINE_ROB_TEST)
TEST_COUNTS(DEFINE_NOP_CONTROL)

// -------- Test configuration --------

typedef double (*rob_test_fn)(
    Struct1 *, Struct1 *, uint64_t
);

typedef double (*nop_control_fn)(uint64_t);

typedef struct {

    int nops;
    rob_test_fn rob_function;
    nop_control_fn control_function;

} TestCase;

#define TEST_ENTRY(N) \
    {N, rob_test_##N, nop_control_##N},

static const TestCase tests[] = {
    TEST_COUNTS(TEST_ENTRY)
};

#undef TEST_ENTRY

// -------- Main function --------

int main(void) {

    size_t n = NODE_COUNT;

    // Allocate two independent 64-byte-aligned chains
    Struct1 *chain_a =
        aligned_alloc(64, n * sizeof(Struct1));

    Struct1 *chain_b =
        aligned_alloc(64, n * sizeof(Struct1));

    if (!chain_a || !chain_b) {

        fprintf(stderr, "Memory allocation failed\n");

        free(chain_a);
        free(chain_b);

        return EXIT_FAILURE;
    }

    // Build independent randomized pointer chains
    make_random_cycle(chain_a, n, 12345u);
    make_random_cycle(chain_b, n, 67890u);

    // -------- Baseline measurement --------

    // Warm up the baseline before measuring
    chase_batch(chain_a, WARMUP_STEPS);

    double baseline = chase_batch(chain_a, STEPS);

    fprintf(stderr,
            "Baseline pointer chasing: %.2f ticks/step\n",
            baseline);

    fprintf(stderr,
            "Starting ROB and NOP control experiments...\n");

    // -------- CSV header --------

    printf("experiment,nops,trial,ticks_per_iteration\n");

    size_t num_tests = sizeof(tests) / sizeof(tests[0]);

    // Warm up all benchmark variants
    for (size_t t = 0; t < num_tests; t++) {

        tests[t].rob_function(
            chain_a, chain_b, WARMUP_STEPS);

        tests[t].control_function(WARMUP_STEPS);
    }

    // -------- Measurement loop --------

    for (int trial = 1; trial <= TRIALS; trial++) {

        // Alternate NOP sweep direction each trial.
        // This reduces systematic bias from always
        // measuring the smallest NOP count first.

        for (size_t k = 0; k < num_tests; k++) {

            size_t t;

            if (trial % 2 == 1) {
                t = k;
            } else {
                t = num_tests - 1 - k;
            }

            int nops = tests[t].nops;

            double rob_ticks;
            double control_ticks;

            // Alternate which experiment runs first
            if (trial % 2 == 1) {

                rob_ticks = tests[t].rob_function(
                    chain_a, chain_b, STEPS);

                control_ticks =
                    tests[t].control_function(STEPS);

            } else {

                control_ticks =
                    tests[t].control_function(STEPS);

                rob_ticks = tests[t].rob_function(
                    chain_a, chain_b, STEPS);
            }

            // ROB experiment result
            printf("rob,%d,%d,%.4f\n",
                   nops, trial, rob_ticks);

            // NOP-only control result
            printf("control,%d,%d,%.4f\n",
                   nops, trial, control_ticks);
        }

        fprintf(stderr,
                "Completed trial %d of %d\n",
                trial, TRIALS);
    }

    // -------- Cleanup --------

    free(chain_a);
    free(chain_b);

    fprintf(stderr, "Experiment completed.\n");

    return EXIT_SUCCESS;
}
