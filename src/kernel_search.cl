/*
 * kernel_search.cl — Cosine similarity search with top-K selection
 * OpenCL 1.2 compatible, uses local memory for top-K tracking
 */

#define MAX_K 16

typedef struct {
    uint  index;
    float score;
} search_result_t;

/*
 * Cosine similarity search against uploaded database vectors.
 * Each work-item computes similarity against a subset of DB vectors,
 * then work-group collaboratively finds top-K via local memory reduction.
 *
 * @param db_vectors  Database vectors [db_count * dim] (already normalized)
 * @param query       Query vector [dim] (already normalized)
 * @param db_count    Number of DB vectors
 * @param dim         Dimensionality
 * @param results     Output top-K results [work_groups * MAX_K]
 */
__kernel void kernel_search(
    __global const float *db_vectors,
    __global const float *query,
    const uint db_count,
    const uint dim,
    __global search_result_t *results)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lws = get_local_size(0);
    uint group_id = get_group_id(0);

    /* Local memory: each thread tracks its personal top-K candidates */
    /* We use a simple approach: each thread computes one dot product,
       then we do a parallel top-K selection in local memory */

    __local float local_scores[256];  /* per-thread best score */
    __local uint  local_indices[256]; /* per-thread best index */
    __local search_result_t local_topk[MAX_K];

    /* Each work-item processes multiple DB vectors (strided) */
    float best_score = -2.0f;
    uint  best_index = 0;

    for (uint i = gid; i < db_count; i += get_global_size(0)) {
        /* Compute dot product (cosine sim for normalized vectors) */
        float sim = 0.0f;
        __global const float *vec = db_vectors + (size_t)i * dim;
        for (uint d = 0; d < dim; d++) {
            sim += vec[d] * query[d];
        }
        if (sim > best_score) {
            best_score = sim;
            best_index = i;
        }
    }

    /* Store in local memory */
    if (lid < 256) {
        local_scores[lid] = best_score;
        local_indices[lid] = best_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Parallel top-K selection: reduction in local memory */
    /* Initialize local top-K */
    if (lid < MAX_K) {
        local_topk[lid].score = -2.0f;
        local_topk[lid].index = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Simple approach: iterate through local scores, maintain top-K */
    uint n_local = min(lws, (uint)256);
    for (uint iter = 0; iter < n_local; iter++) {
        /* Find max among remaining */
        if (lid == 0) {
            float max_score = -2.0f;
            uint  max_idx = 0;
            uint  max_tid = 0;

            for (uint t = 0; t < n_local; t++) {
                if (local_scores[t] > max_score) {
                    max_score = local_scores[t];
                    max_idx = local_indices[t];
                    max_tid = t;
                }
            }

            /* Check if this beats worst in top-K */
            int worst_k = 0;
            float worst_score = local_topk[0].score;
            for (int k = 1; k < MAX_K; k++) {
                if (local_topk[k].score < worst_score) {
                    worst_score = local_topk[k].score;
                    worst_k = k;
                }
            }

            if (max_score > worst_score) {
                local_topk[worst_k].score = max_score;
                local_topk[worst_k].index = max_idx;
                local_scores[max_tid] = -2.0f; /* mark as used */
            } else {
                /* No more improvements possible */
                local_scores[0] = -3.0f; /* signal done */
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Check termination */
        if (local_scores[0] == -3.0f) break;
    }

    /* Write results */
    if (lid < MAX_K) {
        uint out_base = group_id * MAX_K;
        results[out_base + lid] = local_topk[lid];
    }
}

/*
 * Final reduction: merge per-group top-K into global top-K.
 * Called with a single work-group.
 */
__kernel void kernel_search_reduce(
    __global const search_result_t *group_results,
    const uint num_groups,
    __global search_result_t *final_results)
{
    uint lid = get_local_id(0);
    uint lws = get_local_size(0);

    __local search_result_t merge_buf[MAX_K * 32]; /* max 32 groups */
    __local search_result_t final_topk[MAX_K];

    /* Load group results into local memory */
    for (uint g = lid; g < num_groups * MAX_K; g += lws) {
        if (g < num_groups * MAX_K && g < MAX_K * 32) {
            merge_buf[g] = group_results[g];
        }
    }

    if (lid < MAX_K) {
        final_topk[lid].score = -2.0f;
        final_topk[lid].index = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Single-thread top-K selection from all candidates */
    if (lid == 0) {
        uint total = min(num_groups * MAX_K, (uint)(MAX_K * 32));
        for (uint i = 0; i < total; i++) {
            float s = merge_buf[i].score;
            uint  idx = merge_buf[i].index;
            if (s <= -2.0f) continue;

            /* Find worst in current top-K */
            int worst_k = 0;
            float worst_s = final_topk[0].score;
            for (int k = 1; k < MAX_K; k++) {
                if (final_topk[k].score < worst_s) {
                    worst_s = final_topk[k].score;
                    worst_k = k;
                }
            }
            if (s > worst_s) {
                final_topk[worst_k].score = s;
                final_topk[worst_k].index = idx;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Write final results */
    if (lid < MAX_K) {
        final_results[lid] = final_topk[lid];
    }
}
