/*
 * test_opencl.c — Tests for tile-opencl
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_opencl.h"

static int test_passed = 0;
static int test_failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { \
        printf("  PASS: %s\n", msg); \
        test_passed++; \
    } else { \
        printf("  FAIL: %s\n", msg); \
        test_failed++; \
    } \
} while (0)

static int test_init(tile_context_t **ctx) {
    printf("\n--- Test: Device Init ---\n");
    tile_device_info_t info;
    int ret = tile_init(ctx, &info);
    ASSERT(ret == 0, "tile_init succeeds");
    if (ret == 0) {
        ASSERT(info.compute_units > 0, "Has compute units");
        ASSERT(info.global_mem > 0, "Has global memory");
        ASSERT(info.max_workgroup_size > 0, "Has workgroup size");
    }
    return ret;
}

static void test_hash(tile_context_t *ctx) {
    printf("\n--- Test: BLAKE2b Hashing ---\n");

    /* Hash 10 items of 32 bytes each */
    uint32_t count = 10;
    size_t in_len = 32;
    uint8_t input[320];
    uint8_t output[320];

    for (int i = 0; i < 320; i++) input[i] = (uint8_t)(i * 7 + 13);

    int ret = tile_hash_blake2b(ctx, input, in_len, count, output);
    ASSERT(ret == 0, "Hash kernel succeeds");

    /* Verify non-zero output */
    int nonzero = 0;
    for (int i = 0; i < 320; i++) {
        if (output[i] != 0) { nonzero = 1; break; }
    }
    ASSERT(nonzero, "Hash output is non-zero");

    /* Verify determinism: same input -> same output */
    uint8_t output2[320];
    tile_hash_blake2b(ctx, input, in_len, count, output2);
    ASSERT(memcmp(output, output2, 320) == 0, "Hash is deterministic");

    /* Different input -> different output */
    input[0] ^= 0xFF;
    uint8_t output3[320];
    tile_hash_blake2b(ctx, input, in_len, count, output3);
    ASSERT(memcmp(output, output3, 320) != 0, "Different input produces different hash");
}

static void test_embed(tile_context_t *ctx) {
    printf("\n--- Test: Embedding ---\n");

    uint32_t count = 5;
    uint32_t dim = 128;
    size_t in_len = 32;
    uint8_t input[160];
    float embeddings[640];

    for (int i = 0; i < 160; i++) input[i] = (uint8_t)(i + i * 3);

    int ret = tile_embed(ctx, input, in_len, count, embeddings, dim);
    ASSERT(ret == 0, "Embed kernel succeeds");

    /* Check embeddings are unit-normalized */
    for (uint32_t i = 0; i < count; i++) {
        float norm = 0;
        for (uint32_t d = 0; d < dim; d++) {
            float v = embeddings[i * dim + d];
            norm += v * v;
        }
        norm = sqrtf(norm);
        float diff = fabsf(norm - 1.0f);
        ASSERT(diff < 0.01f, "Embedding is unit-normalized");
        if (diff >= 0.01f) {
            printf("    (norm[%u] = %.4f)\n", i, norm);
            break;
        }
    }
}

static void test_search(tile_context_t *ctx) {
    printf("\n--- Test: Search ---\n");

    uint32_t count = 1000;
    uint32_t dim = 128;

    float *db = malloc((size_t)count * dim * sizeof(float));
    float query[128];

    srand(12345);
    for (size_t i = 0; i < (size_t)count * dim; i++) {
        db[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    /* Normalize */
    for (uint32_t i = 0; i < count; i++) {
        float norm = 0;
        for (uint32_t d = 0; d < dim; d++) {
            float v = db[(size_t)i * dim + d];
            norm += v * v;
        }
        norm = sqrtf(norm);
        for (uint32_t d = 0; d < dim; d++)
            db[(size_t)i * dim + d] /= norm;
    }

    /* Use first vector as query — should find itself with score ~1.0 */
    for (uint32_t d = 0; d < dim; d++) query[d] = db[d];

    tile_upload_db(ctx, db, count, dim);

    tile_search_result_t results[TILE_MAX_RESULTS];
    uint32_t n_results;
    int ret = tile_search(ctx, query, dim, results, &n_results);
    ASSERT(ret == 0, "Search kernel succeeds");
    ASSERT(n_results > 0, "Search returns results");

    if (n_results > 0) {
        printf("  Top result: index=%u, score=%.4f\n",
               results[0].index, results[0].score);
        ASSERT(results[0].index == 0, "Finds exact match at index 0");
        ASSERT(results[0].score > 0.99f, "Exact match score ~1.0");
    }

    free(db);
}

static void test_evolve(tile_context_t *ctx) {
    printf("\n--- Test: Evolve ---\n");

    uint32_t count = 100;
    uint32_t dim = 32;
    float *db = malloc((size_t)count * dim * sizeof(float));

    srand(999);
    for (size_t i = 0; i < (size_t)count * dim; i++) {
        db[i] = (float)rand() / (float)RAND_MAX;
    }

    tile_upload_db(ctx, db, count, dim);

    tile_search_result_t evolve_results[4] = {
        {0, 0.9f}, {1, 0.8f}, {2, 0.7f}, {3, 0.6f}
    };

    int ret = tile_evolve(ctx, evolve_results, 4, 0.1f, 0.0f, 2.0f);
    ASSERT(ret == 0, "Evolve kernel succeeds");

    /* Read back scores to verify they changed */
    float *scores = malloc(count * sizeof(float));
    tile_read_scores(ctx, scores, count);

    /* Scores at evolved indices should be != 0.5 */
    int changed = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(scores[evolve_results[i].index] - 0.5f) > 0.001f) {
            changed = 1;
            break;
        }
    }
    ASSERT(changed, "Scores were updated by evolve");

    free(db);
    free(scores);
}

static void test_benchmark(tile_context_t *ctx) {
    printf("\n--- Test: Benchmarks ---\n");

    int ret;
    ret = tile_benchmark(ctx, 1000, 128);
    ASSERT(ret == 0, "Benchmark 1K vectors succeeds");

    ret = tile_benchmark(ctx, 10000, 128);
    ASSERT(ret == 0, "Benchmark 10K vectors succeeds");

    ret = tile_benchmark(ctx, 100000, 128);
    ASSERT(ret == 0, "Benchmark 100K vectors succeeds");
}

int main(int argc, char **argv) {
    printf("=== tile-opencl Test Suite ===\n");

    tile_context_t *ctx = NULL;

    if (test_init(&ctx) != 0) {
        printf("\nFATAL: Cannot initialize OpenCL. Aborting.\n");
        return 1;
    }

    test_hash(ctx);
    test_embed(ctx);
    test_search(ctx);
    test_evolve(ctx);

    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        test_benchmark(ctx);
    } else {
        printf("\n(Use --bench to run benchmarks at 1K/10K/100K vectors)\n");
    }

    tile_cleanup(ctx);

    printf("\n=== Results: %d passed, %d failed ===\n",
           test_passed, test_failed);

    return test_failed > 0 ? 1 : 0;
}
