#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>

/*
    Physical Register File (PRF) size reverse engineering

    Per iteration:
        1. Load from pointer chain A       (cache/TLB miss, long latency)
        2. N filler instructions that each WRITE a register of one class,
           but do not depend on the load
        3. Load from independent pointer chain B

    Why it works:
        Every register-writing instruction needs a free physical register
        at rename and holds it until it RETIRES. Retirement is blocked by
        the missing load in (1). Once the in-flight filler instructions use
        up the free physical registers of that class, rename stalls, load B
        cannot issue under the shadow of load A, and time per iteration
        steps from ~1 miss latency to ~2.

    Classes tested (each filler unit = 8 instructions):
        nop  : takes a ROB entry, NO physical register -> ROB knee (reference)
        gpr  : integer adds (reg,reg)                  -> integer PRF
        simd : paddd xmm                               -> vector/FP PRF
        mix  : 4 gpr + 4 simd interleaved              -> same file or separate?

    Interpretation (see analyze.py):
        knee_gpr / knee_simd < knee_nop : that class is PRF-limited
        PRF size ~= knee + architectural registers held (16 GPR, 16 xmm,
                    32 if AVX-512 hardware) -- the committed arch state
                    occupies physical registers too.
        knee_mix ~= knee_gpr ~= knee_simd -> unified file
        knee_mix ~= 2x the smaller knee   -> separate files (capped by ROB)
*/

#define NODE_COUNT (1024 * 1024)
#define STEPS 100000
#define TRIALS 15
#define WARMUP_STEPS 10000

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

// -------- Filler units (8 instructions each) --------
//
// reg,reg adds are used instead of add-immediate because some cores can
// fold small immediates at rename. r15 / xmm8 are never written, so the
// fillers have no dependence on the load. Destinations rotate over 8
// registers so no single dependency chain gets long.

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
    "add %%r15, %%r8\n\t"     "paddd %%xmm8, %%xmm0\n\t" \
    "add %%r15, %%r9\n\t"     "paddd %%xmm8, %%xmm1\n\t" \
    "add %%r15, %%r10\n\t"    "paddd %%xmm8, %%xmm2\n\t" \
    "add %%r15, %%r11\n\t"    "paddd %%xmm8, %%xmm3\n\t"

// Same clobber list for every class so compiler register allocation of
// the loop is identical across tests.
#define CLOBBERS \
    "rax", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", \
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", \
    "xmm8", "memory", "cc"

// -------- Test generators --------
// REPS = number of 8-instruction units, so filler count = 8 * REPS.

#define DEFINE_PRF_TEST(CLS, UNIT, REPS)                              \
static __attribute__((noinline))                                      \
double CLS##_test_##REPS(Struct1 *a, Struct1 *b, uint64_t steps) {    \
    uint64_t t0 = x86_tsc_start();                                    \
    for (uint64_t i = 0; i < steps; i++) {                            \
        __asm__ volatile (                                            \
            "mov (%0), %0\n\t"                                        \
            ".rept " STRINGIFY(REPS) "\n\t"                           \
            UNIT                                                      \
            ".endr\n\t"                                               \
            "mov (%1), %1\n\t"                                        \
            : "+r"(a), "+r"(b) : : CLOBBERS);                         \
    }                                                                 \
    uint64_t t1 = x86_tsc_stop();                                     \
    sink_a = a;                                                       \
    sink_b = b;                                                       \
    return (double)(t1 - t0) / steps;                                 \
}

#define DEFINE_CONTROL(CLS, UNIT, REPS)                               \
static __attribute__((noinline))                                      \
double CLS##_control_##REPS(uint64_t steps) {                         \
    uint64_t t0 = x86_tsc_start();                                    \
    for (uint64_t i = 0; i < steps; i++) {                            \
        __asm__ volatile (                                            \
            ".rept " STRINGIFY(REPS) "\n\t"                           \
            UNIT                                                      \
            ".endr\n\t"                                               \
            : : : CLOBBERS);                                          \
    }                                                                 \
    uint64_t t1 = x86_tsc_stop();                                     \
    return (double)(t1 - t0) / steps;                                 \
}

// Sweep list: units of 8 instructions (so 2 -> 16 instrs, 80 -> 640).
#define TEST_COUNTS(X) \
    X(0)  X(2)  X(4)  X(6)  X(8)  X(10) X(12) X(14) X(16) X(18) \
    X(20) X(22) X(24) X(26) X(28) X(30) X(32) X(34) X(36) X(38) \
    X(40) X(44) X(48) X(52) X(56) X(60) X(64) X(72) X(80)

#define GEN_ALL(N)                         \
    DEFINE_PRF_TEST(nop,  UNIT_NOP,  N)    \
    DEFINE_PRF_TEST(gpr,  UNIT_GPR,  N)    \
    DEFINE_PRF_TEST(simd, UNIT_SIMD, N)    \
    DEFINE_PRF_TEST(mix,  UNIT_MIX,  N)    \
    DEFINE_CONTROL(nop,   UNIT_NOP,  N)    \
    DEFINE_CONTROL(gpr,   UNIT_GPR,  N)    \
    DEFINE_CONTROL(simd,  UNIT_SIMD, N)    \
    DEFINE_CONTROL(mix,   UNIT_MIX,  N)

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

#define TEST_ENTRY(N) \
    { (N) * 8, \
      { nop_test_##N, gpr_test_##N, simd_test_##N, mix_test_##N }, \
      { nop_control_##N, gpr_control_##N, simd_control_##N, mix_control_##N } },

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

    // Warm up every variant
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
