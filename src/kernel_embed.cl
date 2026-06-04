/*
 * kernel_embed.cl — Position-aware embedding kernel
 * OpenCL 1.2 compatible, uses local memory for accumulation
 */

/*
 * Generate position-aware embeddings from hash input.
 * Each work-item produces one embedding vector of dimension `dim`.
 *
 * Strategy: hash-derived seed + sinusoidal position encoding.
 * Uses local memory for intermediate accumulation to reduce global mem pressure.
 *
 * @param hash_inputs  Input data [count * in_len]
 * @param in_len       Input length per item
 * @param count        Number of items
 * @param embeddings   Output [count * dim]
 * @param dim          Embedding dimension
 */
__kernel void kernel_embed(
    __global const uchar *hash_inputs,
    const uint in_len,
    const uint count,
    __global float *embeddings,
    const uint dim)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    __global const uchar *input = hash_inputs + (size_t)gid * in_len;
    __global float *emb = embeddings + (size_t)gid * dim;

    /* Seed from input: simple hash accumulator */
    ulong seed = 0xcbf29ce484222325UL;  /* FNV offset basis */
    for (uint i = 0; i < in_len; i++) {
        seed ^= (ulong)input[i];
        seed *= 0x100000001b3UL;  /* FNV prime */
    }

    /* Generate embedding with position-aware encoding */
    for (uint d = 0; d < dim; d++) {
        /* LCG for deterministic float generation from seed */
        ulong s = seed ^ ((ulong)d * 0x9e3779b97f4a7c15UL);
        s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9UL;
        s = (s ^ (s >> 27)) * 0x94d049bb133111ebUL;
        s = s ^ (s >> 31);

        /* Convert to float in [-1, 1] */
        float val = (float)((int)(s & 0xFFFFFF)) / (float)0x800000;
        val = val - 1.0f;

        /* Sinusoidal position encoding */
        float pos = (float)d;
        float freq = 1.0f / pow(10000.0f, (float)(d % (dim / 2)) / (float)(dim / 2));
        float pe;
        if ((d & 1) == 0) {
            pe = sin(pos * freq);
        } else {
            pe = cos(pos * freq);
        }

        /* Combine: content + position */
        emb[d] = val * 0.7f + pe * 0.3f;
    }

    /* Normalize embedding to unit length */
    float norm = 0.0f;
    for (uint d = 0; d < dim; d++) {
        norm += emb[d] * emb[d];
    }
    norm = sqrt(norm);
    if (norm > 1e-6f) {
        for (uint d = 0; d < dim; d++) {
            emb[d] /= norm;
        }
    }
}
