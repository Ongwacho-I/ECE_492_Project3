
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>


#define NODE_COUNT (1024 * 1024)
#define STEPS 10000
#define TRIALS 7
#define WARMUP_STEPS 2000

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)


/*
    ROB Reverse Engineering

    1. Write a loop containing a long chain of dependent instructions
       that stall the pipeline.

       Use pointer chasing to create dependent memory accesses.

    2. Follow the long-latency operation with varying numbers of
       independent, fast instructions.

    3. Measure execution time.

       Independent instructions may execute out of order while
       an older long-latency instruction is waiting.

    4. When the available ROB capacity is exceeded, instruction
       allocation may stall, potentially affecting execution time.

    Precise Exception Verification:

       Trigger a fault and investigate whether younger instructions
       retire architecturally. Precise exceptions prevent younger
       instructions from committing architectural state, even if
       they executed speculatively.
*/


// Step 1: Pointer chasing to create a long dependency chain


// -------- Create timing primitives --------

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


// -------- Create pointer-chasing structs --------

typedef struct node {

    struct node *next;
	unsigned char padding[64 - sizeof(void *)];
} Struct1;


// Prevent compiler from eliminating the final pointer result
static Struct1 * volatile sink_a;
static Struct1 * volatile sink_b;



// -------- Xorshift32 random number generator --------

static uint32_t xorshift32(uint32_t *state) {

    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return *state = x;

} // End of xorshift32


// -------- Randomize pointer-chasing cycle --------

// Each node's next pointer points to a shuffled successor

static void make_random_cycle(Struct1 *nodes, size_t n, uint32_t seed) {

    if (n == 0 || n > SIZE_MAX / sizeof(size_t) ||
        n > UINT32_MAX) {

        fprintf(stderr, "make_random_cycle size error\n");
        exit(EXIT_FAILURE);

    }

    size_t *order = malloc(n * sizeof(*order));

    if (!order) {

        fprintf(stderr, "make_random_cycle allocation error\n");
        exit(EXIT_FAILURE);

    } // End of if statement

    uint32_t state = seed ? seed : 1u;


    // Initialize order array
    for (size_t i = 0; i < n; i++) {

        order[i] = i;

    } // End of for loop


    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {

        size_t j = (size_t)(xorshift32(&state) %
                            (uint32_t)(i + 1));

        size_t tmp = order[i];

        order[i] = order[j];

        order[j] = tmp;

    } // End of shuffle loop


    // Connect nodes into one randomized circular chain
    for (size_t i = 0; i < n; i++) {

        nodes[order[i]].next =
            &nodes[order[(i + 1) % n]];

    } // End of pointer connection loop

    free(order);

} // End of make_random_cycle


//-------------Baseline pointer-chsing function
static double chase_batch( Struct1 *p, uint64_t steps ) {
    uint64_t t0 = x86_tsc_start();

    for (uint64_t i = 0; i < steps; i++) {
        p = p->next;
    }

    uint64_t t1 = x86_tsc_stop();

    sink_a = p;

    return (double)(t1 - t0);
}

//----------ROB-capacity measurment---------------------

// Generate a separate function for each NOP count.
//
// Each iteration:
// 1. Load next pointer in chain A.
// 2. Execute N NOP instructions.
// 3. Load next pointer in chain B.
//
// Both chains are independent.

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




// -------- Generate benchmark variants --------

DEFINE_ROB_TEST(0)
DEFINE_ROB_TEST(16)
DEFINE_ROB_TEST(32)
DEFINE_ROB_TEST(64)
DEFINE_ROB_TEST(96)
DEFINE_ROB_TEST(128)
DEFINE_ROB_TEST(160)
DEFINE_ROB_TEST(192)
DEFINE_ROB_TEST(224)
DEFINE_ROB_TEST(256)
DEFINE_ROB_TEST(320)
DEFINE_ROB_TEST(384)
DEFINE_ROB_TEST(448)
DEFINE_ROB_TEST(512)








// -------- Test configuration --------

typedef double (*rob_test_fn)(
    Struct1 *, Struct1 *, uint64_t
);

typedef struct {
    int nops;
    rob_test_fn function;
} TestCase;

static const TestCase tests[] = {
    {0,   rob_test_0},
    {16,  rob_test_16},
    {32,  rob_test_32},
    {64,  rob_test_64},
    {96,  rob_test_96},
    {128, rob_test_128},
    {160, rob_test_160},
    {192, rob_test_192},
    {224, rob_test_224},
    {256, rob_test_256},
    {320, rob_test_320},
    {384, rob_test_384},
    {448, rob_test_448},
    {512, rob_test_512}
};













// -------- Main function --------

int main(void) {

    size_t n = NODE_COUNT;

    // Allocate two independent chains
Struct1 *chain_a = aligned_alloc(64, n * sizeof(Struct1));
Struct1 *chain_b = aligned_alloc(64, n * sizeof(Struct1));

    if (!chain_a || !chain_b) {
        fprintf(stderr, "Memory allocation failed\n");
        free(chain_a);
        free(chain_b);
        return EXIT_FAILURE;
    }

    // Build randomized cycles
    make_random_cycle(chain_a, n, 12345u);
    make_random_cycle(chain_b, n, 67890u);

    // Baseline pointer-chasing test
    double baseline = chase_batch(chain_a, STEPS);

    fprintf(stderr,
            "Baseline pointer chasing: %.2f ticks/step\n",
            baseline / STEPS);

    fprintf(stderr,
            "ROB experiment starting...\n");

    // CSV output
    printf("nops,trial,ticks_per_iteration\n");

    size_t num_tests = sizeof(tests) / sizeof(tests[0]);

    // Sweep NOP counts
    for (size_t t = 0; t < num_tests; t++) {

        int nops = tests[t].nops;
        rob_test_fn run = tests[t].function;

        // Warm up each benchmark variant
        run(chain_a, chain_b, WARMUP_STEPS);

        // Repeat measurements
        for (int trial = 1; trial <= TRIALS; trial++) {

            double ticks =
                run(chain_a, chain_b, STEPS);

            printf("%d,%d,%.4f\n",
                   nops, trial, ticks);
        }
    }

    free(chain_a);
    free(chain_b);

    return EXIT_SUCCESS;
}
