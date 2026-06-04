#ifndef TILE_OPENCL_H
#define TILE_OPENCL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration */
#define TILE_VEC_DIM        128
#define TILE_MAX_RESULTS     16
#define TILE_BLAKE2B_OUTLEN  32

/* Device info */
typedef struct {
    char    name[256];
    char    vendor[256];
    char    version[128];
    uint32_t compute_units;
    uint64_t global_mem;
    size_t  max_workgroup_size;
    uint64_t local_mem_size;
    int     is_gpu;
} tile_device_info_t;

/* Search result */
typedef struct {
    uint32_t index;
    float    score;      /* cosine similarity */
} tile_search_result_t;

/* Context handle (opaque outside the impl) */
typedef struct tile_context tile_context_t;

/* ---- API ---- */

/**
 * Initialize OpenCL context. Auto-detects best device (GPU preferred, CPU fallback).
 * Returns 0 on success, negative on error.
 */
int tile_init(tile_context_t **ctx, tile_device_info_t *out_info);

/**
 * Shut down and release all resources.
 */
void tile_cleanup(tile_context_t *ctx);

/**
 * Upload a database of vectors to the device.
 * @param vectors  Host pointer, row-major [count * dim]
 * @param count    Number of vectors
 * @param dim      Dimensionality of each vector
 */
int tile_upload_db(tile_context_t *ctx, const float *vectors,
                   uint32_t count, uint32_t dim);

/**
 * Batch BLAKE2b hash on device.
 * @param input   Host input data (raw bytes)
 * @param in_len  Length of each input element (bytes)
 * @param count   Number of inputs
 * @param output  Host output buffer [count * TILE_BLAKE2B_OUTLEN]
 */
int tile_hash_blake2b(tile_context_t *ctx,
                      const uint8_t *input, size_t in_len, uint32_t count,
                      uint8_t *output);

/**
 * Compute position-aware embeddings on device.
 * @param hash_input  Hash input data [count * in_len]
 * @param in_len      Input length per item
 * @param count       Number of items
 * @param embeddings  Output [count * dim]
 * @param dim         Embedding dimension
 */
int tile_embed(tile_context_t *ctx,
               const uint8_t *hash_input, size_t in_len, uint32_t count,
               float *embeddings, uint32_t dim);

/**
 * Cosine similarity search: find top-K closest vectors.
 * @param query    Query vector [dim]
 * @param dim      Dimensionality
 * @param results  Output array [TILE_MAX_RESULTS]
 * @param n_results  Number of results actually returned
 */
int tile_search(tile_context_t *ctx,
                const float *query, uint32_t dim,
                tile_search_result_t *results, uint32_t *n_results);

/**
 * Evolve scores based on search results with learning rate and clamping.
 * @param results    Search results
 * @param n_results  Number of results
 * @param lr         Learning rate
 * @param clamp_min  Minimum score clamp
 * @param clamp_max  Maximum score clamp
 */
int tile_evolve(tile_context_t *ctx,
                const tile_search_result_t *results, uint32_t n_results,
                float lr, float clamp_min, float clamp_max);

/**
 * Read back scores buffer from device.
 * @param scores  Output buffer [count]
 * @param count   Number of score entries to read
 */
int tile_read_scores(tile_context_t *ctx, float *scores, uint32_t count);

/**
 * Run benchmark: compare OpenCL vs CPU for given vector count.
 * Prints results to stdout.
 */
int tile_benchmark(tile_context_t *ctx,
                   uint32_t count, uint32_t dim);

#ifdef __cplusplus
}
#endif

#endif /* TILE_OPENCL_H */
