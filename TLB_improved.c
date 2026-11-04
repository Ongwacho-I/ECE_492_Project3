#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

/*
 * TLB experiment: randomized dependent loads, one per virtual page.
 * 4K mode probes ordinary data translations. 2M mode is exploratory:
 * MADV_HUGEPAGE is only a request, so inspect the backing information.
 *
 * Usage:
 *   ./TLB_improved [page_bytes [max_pages]]
 *   ./TLB_improved 4096 16384
 *   ./TLB_improved 2097152 128
 *   ./TLB_improved 4096 4096 16 256 4   (linear sweep: start end step)
 *
 * CSV to stdout; diagnostics to stderr.
 * The control is a *cache-sensitive comparison*, not a pure TLB-cost
 * subtraction: memory layout, cache conflict, and prefetch differ.
 */

#ifndef TRIALS
#define TRIALS 9
#endif
#ifndef MIN_ACCESSES
#define MIN_ACCESSES 200000
#endif
#ifndef WARMUP_ACCESSES
#define WARMUP_ACCESSES 20000
#endif

#define PAGE_4K ((size_t)4096)
#define PAGE_2M ((size_t)2097152)
#define CACHELINE ((size_t)64)

static void *volatile sink;

static void die(const char *msg) { perror(msg); exit(EXIT_FAILURE); }
static size_t parse_size(const char *s, const char *what) {
    char *end = NULL;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || !s[0] || !end || *end || n == 0 || n > SIZE_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", what, s);
        exit(EXIT_FAILURE);
    }
    return (size_t)n;
}
static void *checked_malloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) die("malloc");
    return p;
}

static inline uint64_t tsc_start(void) {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}
static inline uint64_t tsc_stop(void) {
    unsigned aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}
static void shuffle(size_t *a, size_t n, uint32_t seed) {
    uint32_t state = seed ? seed : 1;
    for (size_t i = n; i > 1; --i) {
        size_t j = xorshift32(&state) % i;
        size_t tmp = a[i - 1]; a[i - 1] = a[j]; a[j] = tmp;
    }
}

/* Allocate exactly 'bytes' mapped bytes aligned to 'alignment'.
 * Unmap leading/trailing padding so the returned pointer can be munmap'd.
 */
static void *map_aligned(size_t bytes, size_t alignment) {
    if (bytes == 0 || bytes % PAGE_4K || alignment % PAGE_4K ||
        (alignment & (alignment - 1)) || bytes > SIZE_MAX - alignment) {
        fprintf(stderr, "Invalid allocation size/alignment\n");
        exit(EXIT_FAILURE);
    }
    size_t total = bytes + alignment;
    void *raw = mmap(NULL, total, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) die("mmap");
    uintptr_t lo = (uintptr_t)raw;
    uintptr_t aligned = (lo + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t prefix = aligned - lo;
    size_t suffix = total - prefix - bytes;
    if (prefix && munmap(raw, prefix)) die("munmap prefix");
    if (suffix && munmap((void *)(aligned + bytes), suffix)) die("munmap suffix");
    return (void *)aligned;
}

/* Find backing information for the *specific* mapping, when smaps is
 * available (it may be absent or virtualized under Gramine/SGX).
 * AnonHugePages is more useful than KernelPageSize for checking THP.
 */
static long mapping_anon_huge_kb(const void *address) {
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return -1;
    char line[512];
    uintptr_t target = (uintptr_t)address;
    int inside = 0;
    long value = -1;
    while (fgets(line, sizeof line, f)) {
        unsigned long a, b;
        if (sscanf(line, "%lx-%lx", &a, &b) == 2 && strchr(line, '-')) {
            inside = (target >= (uintptr_t)a && target < (uintptr_t)b);
            continue;
        }
        if (inside && sscanf(line, "AnonHugePages: %ld kB", &value) == 1) break;
    }
    fclose(f);
    return value;
}

static inline void **test_node(char *base, size_t page_bytes, size_t p) {
    /* Rotate the offset to avoid all nodes sharing the same L1D set. */
    return (void **)(base + p * page_bytes + (p % 64) * CACHELINE);
}
static inline void **control_node(char *base, size_t p) {
    return (void **)(base + p * CACHELINE);
}
static void build_cycle(char *base, size_t page_bytes, size_t n,
                        uint32_t seed, int control) {
    size_t *order = checked_malloc(n * sizeof(*order));
    for (size_t i = 0; i < n; ++i) order[i] = i;
    if (n > 1) shuffle(order, n, seed);
    for (size_t i = 0; i < n; ++i) {
        size_t cur = order[i], next = order[(i + 1) % n];
        void **at = control ? control_node(base, cur) : test_node(base, page_bytes, cur);
        void **to = control ? control_node(base, next) : test_node(base, page_bytes, next);
        *at = to;
    }
    free(order);
}

static __attribute__((noinline)) double chase(void *p, uint64_t steps) {
    uint64_t begin = tsc_start();
    for (uint64_t i = 0; i < steps; i += 8) {
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
        p = *(void *volatile *)p;
    }
    uint64_t end = tsc_stop();
    sink = p;
    return (double)(end - begin) / (double)steps;
}
static uint64_t round8(uint64_t x) { return (x + 7) & ~(uint64_t)7; }

static size_t *geometric_sweep(size_t max_pages, size_t *out_n) {
    size_t cap = 256, n = 0, previous = 0;
    size_t *list = checked_malloc(cap * sizeof(*list));
    double x = 1.0;
    while (x <= (double)max_pages + 0.49) {
        size_t p = (size_t)(x + 0.5);
        if (p > max_pages) break;
        if (p > previous) {
            if (n == cap) {
                cap *= 2;
                size_t *temp = realloc(list, cap * sizeof(*list));
                if (!temp) die("realloc");
                list = temp;
            }
            list[n++] = p;
            previous = p;
        }
        x *= 1.0594630943592953; /* ~12 points per doubling */
    }
    if (previous != max_pages) {
        if (n == cap) {
            size_t *temp = realloc(list, (cap + 1) * sizeof(*list));
            if (!temp) die("realloc");
            list = temp;
        }
        list[n++] = max_pages;
    }
    *out_n = n;
    return list;
}

int main(int argc, char **argv) {
    size_t page_bytes = argc > 1 ? parse_size(argv[1], "page size") : PAGE_4K;
    if (page_bytes != PAGE_4K && page_bytes != PAGE_2M) {
        fprintf(stderr, "Page size must be 4096 or 2097152\n");
        return EXIT_FAILURE;
    }
    size_t max_pages = argc > 2 ? parse_size(argv[2], "max pages") :
                       (page_bytes == PAGE_4K ? 16384 : 128);
    if (max_pages > SIZE_MAX / page_bytes || max_pages > SIZE_MAX / CACHELINE) {
        fprintf(stderr, "Requested mapping is too large\n");
        return EXIT_FAILURE;
    }
    size_t count = 0;
    size_t *sweep = NULL;
    if (argc == 6) {
        size_t lo = parse_size(argv[3], "start"), hi = parse_size(argv[4], "end");
        size_t step = parse_size(argv[5], "step");
        if (hi < lo || hi > max_pages) {
            fprintf(stderr, "Invalid sweep bounds (end must be <= max_pages)\n");
            return EXIT_FAILURE;
        }
        count = (hi - lo) / step + 1;
        sweep = checked_malloc(count * sizeof(*sweep));
        for (size_t i = 0; i < count; ++i) sweep[i] = lo + i * step;
    } else if (argc <= 3) {
        sweep = geometric_sweep(max_pages, &count);
    } else {
        fprintf(stderr, "Usage: %s [4096|2097152 [max_pages [start end step]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t test_bytes = max_pages * page_bytes;
    size_t ctrl_bytes = max_pages * CACHELINE;
    size_t ctrl_map_bytes = (ctrl_bytes + PAGE_4K - 1) & ~(PAGE_4K - 1);
    char *test = map_aligned(test_bytes, PAGE_2M);
    char *ctrl = map_aligned(ctrl_map_bytes, PAGE_2M);

    int advice = page_bytes == PAGE_4K ? MADV_NOHUGEPAGE : MADV_HUGEPAGE;
    if (madvise(test, test_bytes, advice))
        fprintf(stderr, "WARNING: test madvise failed: %s\n", strerror(errno));
    if (madvise(ctrl, ctrl_map_bytes, MADV_HUGEPAGE))
        fprintf(stderr, "WARNING: control madvise failed: %s\n", strerror(errno));

    /* Force physical backing so the timed chase does not fault in pages. */
    for (size_t i = 0; i < max_pages; ++i)
        test[i * page_bytes] = 0;
    for (size_t i = 0; i < ctrl_map_bytes; i += PAGE_4K)
        ctrl[i] = 0;

    long huge_kb = mapping_anon_huge_kb(test);
    fprintf(stderr, "page_bytes=%zu max_pages=%zu points=%zu trials=%d ",
            page_bytes, max_pages, count, TRIALS);
    if (huge_kb < 0) {
        fprintf(stderr, "test AnonHugePages=unavailable\n");
        if (page_bytes == PAGE_2M)
            fprintf(stderr, "WARNING: 2M backing cannot be verified; do not infer a huge-page TLB size.\n");
    } else {
        fprintf(stderr, "test AnonHugePages=%ld kB\n", huge_kb);
        if (page_bytes == PAGE_2M && (uint64_t)huge_kb * 1024 < test_bytes)
            fprintf(stderr, "WARNING: Some test memory is not verified as THP-backed.\n");
        if (page_bytes == PAGE_4K && huge_kb)
            fprintf(stderr, "WARNING: 4K test appears to contain huge-page backing.\n");
    }

    printf("page_bytes,pages,trial,ticks_per_access,ctrl_ticks_per_access\n");
    for (size_t s = 0; s < count; ++s) {
        size_t pages = sweep[s];
        build_cycle(test, page_bytes, pages, 12345u + (uint32_t)pages, 0);
        build_cycle(ctrl, page_bytes, pages, 777u + (uint32_t)pages, 1);
        void *tp = test_node(test, page_bytes, 0);
        void *cp = control_node(ctrl, 0);
        uint64_t warm = round8((uint64_t)WARMUP_ACCESSES > 2 * (uint64_t)pages ?
                              (uint64_t)WARMUP_ACCESSES : 2 * (uint64_t)pages);
        uint64_t steps = round8((uint64_t)MIN_ACCESSES > 8 * (uint64_t)pages ?
                               (uint64_t)MIN_ACCESSES : 8 * (uint64_t)pages);
        (void)chase(tp, warm);
        (void)chase(cp, warm);
        for (int trial = 1; trial <= TRIALS; ++trial) {
            double t, c;
            if (trial % 2) {
                t = chase(tp, steps); c = chase(cp, steps);
            } else {
                c = chase(cp, steps); t = chase(tp, steps);
            }
            printf("%zu,%zu,%d,%.4f,%.4f\n", page_bytes, pages, trial, t, c);
        }
        if (s % 10 == 0) fprintf(stderr, "point %zu/%zu (pages=%zu)\n", s + 1, count, pages);
    }
    fflush(stdout);
    fprintf(stderr, "Experiment completed.\n");
    if (munmap(test, test_bytes)) perror("munmap test");
    if (munmap(ctrl, ctrl_map_bytes)) perror("munmap control");
    free(sweep);
    return EXIT_SUCCESS;
}
