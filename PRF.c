#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>

/*
    Physical Register File (PRF) size reverse engineering (v2)

    Per iteration:
        1. Load from pointer chain A       (cache/TLB miss, long latency)
        2. N filler instructions that each WRITE a register of one class,
           but do not depend on the load
        3. Load from independent pointer chain B

    Each filler instruction needs a free physical register at rename and
    holds it until it retires. Retirement is blocked by the missing load
    in (1). When the in-flight fillers use up the free registers of that
    class, rename stalls and load B can no longer overlap with load A.

    Classes:
        nop  : ROB entry only, no physical register   -> ROB size reference
        gpr  : integer adds                           -> integer PRF
        simd : paddd xmm                              -> vector/FP PRF
        mix  : 4 gpr + 4 simd interleaved             -> shared vs separate

    v2 changes:
        * Sweep resolution is 4 instructions between 128 and 320, where the
          knees were observed; coarse steps elsewhere.
        * 21 trials by default.
        * Parameters can be overridden: gcc -DTRIALS=31 -DSTEPS=200000 ...

    Filler count = 8*R + 4*T. R = 8-instruction units, T = optional
    4-instruction tail (0 or 1).

    Build: gcc -O2 -Wall -Wextra -std=gnu11 -fno-unroll-loops PRF.c -o PRF
    Run:   taskset -c 10 ./PRF > prf_results.csv
*/

#ifndef NODE_COUNT
#define NODE_COUNT (1024 * 1024)
#endif
#ifndef STEPS
#define STEPS 100000
#endif
#ifndef TRIALS
#define TRIALS 21
#endif
#ifndef WARMUP_STEPS
#define WARMUP_STEPS 10000
#endif

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

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
    unsigned char padding[64 - sizeof(void *)];
} Struct1;

_Static_assert(sizeof(Struct1) == 64, "Node size must be 64 bytes");

static Struct1 * volatile sink_a;
static Struct1 * volatile sink_b;

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

static void make_random_cycle(Struct1 *nodes, size_t n, uint32_t seed) {
    if (n == 0 || n > UINT32_MAX) {
        fprintf(stderr, "Invalid node count\n");
        exit(EXIT_FAILURE);
    }
    size_t *order = malloc(n * sizeof(*order));
    if (!order) {
        fprintf(stderr, "Allocation failed\n");
        exit(EXIT_FAILURE);
    }
    uint32_t state = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) order[i] = i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift32(&state) % (uint32_t)(i + 1));
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
    for (size_t i = 0; i < n; i++)
        nodes[order[i]].next = &nodes[order[(i + 1) % n]];
    free(order);
}

static double chase_batch(Struct1 *p, uint64_t steps) {
    uint64_t t0 = x86_tsc_start();
    for (uint64_t i = 0; i < steps; i++) p = p->next;
    uint64_t t1 = x86_tsc_stop();
    sink_a = p;
    return (double)(t1 - t0) / steps;
}

// -------- Filler units --------
//
// reg,reg adds (not add-immediate) so rename-time immediate folding cannot
// hide the register allocation. r15 / xmm8 are never written, so fillers
// do not depend on the load. Destinations rotate to avoid long chains.

// 8-instruction units
#define UNIT_NOP \
    "nop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t"

#define UNIT_GPR \
    "add %%r15, %%r8\n\t"  "add %%r15, %%r9\n\t"  \
    "add %%r15, %%r10\n\t" "add %%r15, %%r11\n\t" \
    "add %%r15, %%r12\n\t" "add %%r15, %%r13\n\t" \
    "add %%r15, %%r14\n\t" "add %%r15, %%rax\n\t"

#define UNIT_SIMD \
    "paddd %%xmm8, %%xmm0\n\t" "paddd %%xmm8, %%xmm1\n\t" \
    "paddd %%xmm8, %%xmm2\n\t" "paddd %%xmm8, %%xmm3\n\t" \
    "paddd %%xmm8, %%xmm4\n\t" "paddd %%xmm8, %%xmm5\n\t" \
    "paddd %%xmm8, %%xmm6\n\t" "paddd %%xmm8, %%xmm7\n\t"

#define UNIT_MIX \
    "add %%r15, %%r8\n\t"  "paddd %%xmm8, %%xmm0\n\t" \
    "add %%r15, %%r9\n\t"  "paddd %%xmm8, %%xmm1\n\t" \
    "add %%r15, %%r10\n\t" "paddd %%xmm8, %%xmm2\n\t" \
    "add %%r15, %%r11\n\t" "paddd %%xmm8, %%xmm3\n\t"

// 4-instruction tails (used 0 or 1 times for 4-instruction resolution)
#define TAIL_NOP  "nop\n\tnop\n\tnop\n\tnop\n\t"

#define TAIL_GPR \
    "add %%r15, %%r8\n\t"  "add %%r15, %%r9\n\t"  \
    "add %%r15, %%r10\n\t" "add %%r15, %%r11\n\t"

#define TAIL_SIMD \
    "paddd %%xmm8, %%xmm0\n\t" "paddd %%xmm8, %%xmm1\n\t" \
    "paddd %%xmm8, %%xmm2\n\t" "paddd %%xmm8, %%xmm3\n\t"

#define TAIL_MIX \
    "add %%r15, %%r8\n\t" "paddd %%xmm8, %%xmm0\n\t" \
    "add %%r15, %%r9\n\t" "paddd %%xmm8, %%xmm1\n\t"

// Same clobber list for every class so loop register allocation is the
// same across all tests.
#define CLOBBERS \
    "rax", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", \
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", \
    "xmm8", "memory", "cc"

// -------- Test generators --------

#define DEFINE_PRF_TEST(CLS, UNIT, TAIL, R, T)                             \
static __attribute__((noinline))                                           \
double CLS##_test_##R##_##T(Struct1 *a, Struct1 *b, uint64_t steps) {      \
    uint64_t t0 = x86_tsc_start();                                         \
    for (uint64_t i = 0; i < steps; i++) {                                 \
        __asm__ volatile (                                                 \
            "mov (%0), %0\n\t"                                             \
            ".rept " STRINGIFY(R) "\n\t" UNIT ".endr\n\t"                  \
            ".rept " STRINGIFY(T) "\n\t" TAIL ".endr\n\t"                  \
            "mov (%1), %1\n\t"                                             \
            : "+r"(a), "+r"(b) : : CLOBBERS);                              \
    }                                                                      \
    uint64_t t1 = x86_tsc_stop();                                          \
    sink_a = a;                                                            \
    sink_b = b;                                                            \
    return (double)(t1 - t0) / steps;                                      \
}

#define DEFINE_CONTROL(CLS, UNIT, TAIL, R, T)                              \
static __attribute__((noinline))                                           \
double CLS##_control_##R##_##T(uint64_t steps) {                           \
    uint64_t t0 = x86_tsc_start();                                         \
    for (uint64_t i = 0; i < steps; i++) {                                 \
        __asm__ volatile (                                                 \
            ".rept " STRINGIFY(R) "\n\t" UNIT ".endr\n\t"                  \
            ".rept " STRINGIFY(T) "\n\t" TAIL ".endr\n\t"                  \
            : : : CLOBBERS);                                               \
    }                                                                      \
    uint64_t t1 = x86_tsc_stop();                                          \
    return (double)(t1 - t0) / steps;                                      \
}

/*
   Sweep points X(R,T) -> 8*R + 4*T instructions:
     coarse   0..112 step 16
     fine     128..320 step 4   (R=16..39 with T=0/1, then 40,0)
     coarse   352,384,416,448,512,576,640
*/
#define TEST_COUNTS(X) \
    X(0,0)  X(2,0)  X(4,0)  X(6,0)  X(8,0)  X(10,0) X(12,0) X(14,0) \
    X(16,0) X(16,1) X(17,0) X(17,1) X(18,0) X(18,1) X(19,0) X(19,1) \
    X(20,0) X(20,1) X(21,0) X(21,1) X(22,0) X(22,1) X(23,0) X(23,1) \
    X(24,0) X(24,1) X(25,0) X(25,1) X(26,0) X(26,1) X(27,0) X(27,1) \
    X(28,0) X(28,1) X(29,0) X(29,1) X(30,0) X(30,1) X(31,0) X(31,1) \
    X(32,0) X(32,1) X(33,0) X(33,1) X(34,0) X(34,1) X(35,0) X(35,1) \
    X(36,0) X(36,1) X(37,0) X(37,1) X(38,0) X(38,1) X(39,0) X(39,1) \
    X(40,0) \
    X(44,0) X(48,0) X(52,0) X(56,0) X(64,0) X(72,0) X(80,0)

#define GEN_ALL(R, T)                                  \
    DEFINE_PRF_TEST(nop,  UNIT_NOP,  TAIL_NOP,  R, T)  \
    DEFINE_PRF_TEST(gpr,  UNIT_GPR,  TAIL_GPR,  R, T)  \
    DEFINE_PRF_TEST(simd, UNIT_SIMD, TAIL_SIMD, R, T)  \
    DEFINE_PRF_TEST(mix,  UNIT_MIX,  TAIL_MIX,  R, T)  \
    DEFINE_CONTROL(nop,   UNIT_NOP,  TAIL_NOP,  R, T)  \
    DEFINE_CONTROL(gpr,   UNIT_GPR,  TAIL_GPR,  R, T)  \
    DEFINE_CONTROL(simd,  UNIT_SIMD, TAIL_SIMD, R, T)  \
    DEFINE_CONTROL(mix,   UNIT_MIX,  TAIL_MIX,  R, T)

TEST_COUNTS(GEN_ALL)

// -------- Test table --------

typedef double (*prf_fn)(Struct1 *, Struct1 *, uint64_t);
typedef double (*ctl_fn)(uint64_t);

#define NUM_CLASSES 4
static const char *class_names[NUM_CLASSES] = {"nop", "gpr", "simd", "mix"};

typedef struct {
    int instrs;
    prf_fn test[NUM_CLASSES];
    ctl_fn control[NUM_CLASSES];
} TestCase;

#define TEST_ENTRY(R, T) \
    { (R) * 8 + (T) * 4, \
      { nop_test_##R##_##T, gpr_test_##R##_##T, \
        simd_test_##R##_##T, mix_test_##R##_##T }, \
      { nop_control_##R##_##T, gpr_control_##R##_##T, \
        simd_control_##R##_##T, mix_control_##R##_##T } },

static const TestCase tests[] = { TEST_COUNTS(TEST_ENTRY) };
#undef TEST_ENTRY

// -------- Main --------

int main(void) {
    size_t n = NODE_COUNT;

    Struct1 *chain_a = aligned_alloc(64, n * sizeof(Struct1));
    Struct1 *chain_b = aligned_alloc(64, n * sizeof(Struct1));
    if (!chain_a || !chain_b) {
        fprintf(stderr, "Memory allocation failed\n");
        free(chain_a);
        free(chain_b);
        return EXIT_FAILURE;
    }

    make_random_cycle(chain_a, n, 12345u);
    make_random_cycle(chain_b, n, 67890u);

    chase_batch(chain_a, WARMUP_STEPS);
    double baseline = chase_batch(chain_a, STEPS);
    fprintf(stderr, "Baseline pointer chasing: %.2f ticks/step\n", baseline);

    printf("class,kind,instrs,trial,ticks_per_iteration\n");

    size_t num_tests = sizeof(tests) / sizeof(tests[0]);
    fprintf(stderr, "%zu sweep points x %d classes x %d trials\n",
            num_tests, NUM_CLASSES, TRIALS);

    for (size_t t = 0; t < num_tests; t++)
        for (int c = 0; c < NUM_CLASSES; c++) {
            tests[t].test[c](chain_a, chain_b, WARMUP_STEPS);
            tests[t].control[c](WARMUP_STEPS);
        }

    for (int trial = 1; trial <= TRIALS; trial++) {
        for (size_t k = 0; k < num_tests; k++) {
            // Alternate sweep direction to reduce ordering bias
            size_t t = (trial % 2 == 1) ? k : num_tests - 1 - k;

            for (int ci = 0; ci < NUM_CLASSES; ci++) {
                int c = (trial % 2 == 1) ? ci : NUM_CLASSES - 1 - ci;
                double rt, ct;

                if (trial % 2 == 1) {
                    rt = tests[t].test[c](chain_a, chain_b, STEPS);
                    ct = tests[t].control[c](STEPS);
                } else {
                    ct = tests[t].control[c](STEPS);
                    rt = tests[t].test[c](chain_a, chain_b, STEPS);
                }

                printf("%s,test,%d,%d,%.4f\n",
                       class_names[c], tests[t].instrs, trial, rt);
                printf("%s,control,%d,%d,%.4f\n",
                       class_names[c], tests[t].instrs, trial, ct);
            }
        }
        fprintf(stderr, "Completed trial %d of %d\n", trial, TRIALS);
    }

    free(chain_a);
    free(chain_b);
    fprintf(stderr, "Experiment completed.\n");
    return EXIT_SUCCESS;
}
