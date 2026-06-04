/*
 * kernel_evolve.cl — Score evolution with atomic updates
 * OpenCL 1.2 compatible
 */

/*
 * Evolve (update) scores based on search results.
 * Uses atomic operations for safe concurrent updates.
 *
 * @param scores       Score array [count]
 * @param indices      Result indices to update [n_results]
 * @param similarities Result similarities [n_results]
 * @param n_results    Number of results
 * @param lr           Learning rate
 * @param clamp_min    Minimum clamp value
 * @param clamp_max    Maximum clamp value
 * @param count        Total number of scores
 */
__kernel void kernel_evolve(
    __global float *scores,
    __global const uint *indices,
    __global const float *similarities,
    const uint n_results,
    const float lr,
    const float clamp_min,
    const float clamp_max,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid >= n_results) return;

    uint idx = indices[gid];
    if (idx >= count) return;

    float sim = similarities[gid];
    float delta = lr * sim;

    /* Atomic update: read-modify-write with clamp */
    /* OpenCL 1.2 doesn't have atomic_add for floats,
       so we use atomic_cmpxchg on the int representation */
    union { float f; uint u; } old_val, new_val, cur_val;

    int done = 0;
    for (int attempts = 0; attempts < 64 && !done; attempts++) {
        old_val.f = scores[idx];
        new_val.f = old_val.f + delta;

        /* Clamp */
        if (new_val.f < clamp_min) new_val.f = clamp_min;
        if (new_val.f > clamp_max) new_val.f = clamp_max;

        cur_val.u = atomic_cmpxchg(
            (__global uint*)&scores[idx],
            old_val.u,
            new_val.u
        );
        done = (cur_val.u == old_val.u);
    }
}

/*
 * Batch score evolution: apply uniform decay + boost to all scores.
 * Simple parallel kernel, one score per work-item.
 *
 * @param scores    Score array [count]
 * @param count     Number of scores
 * @param decay     Decay factor (multiplied, e.g. 0.999)
 * @param boost     Additive boost (applied after decay)
 * @param clamp_min Minimum value
 * @param clamp_max Maximum value
 */
__kernel void kernel_evolve_decay(
    __global float *scores,
    const uint count,
    const float decay,
    const float boost,
    const float clamp_min,
    const float clamp_max)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    float s = scores[gid];
    s = s * decay + boost;
    if (s < clamp_min) s = clamp_min;
    if (s > clamp_max) s = clamp_max;
    scores[gid] = s;
}
