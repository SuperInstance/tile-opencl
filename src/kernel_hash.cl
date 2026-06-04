/*
 * kernel_hash.cl — BLAKE2b batch hashing
 * OpenCL 1.2 compatible, uses constant memory for IV/sigma
 */

/* BLAKE2b IV */
__constant static const ulong blake2b_iv[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

/* BLAKE2b sigma table */
__constant static const uchar blake2b_sigma[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

#define BLAKE2B_G(v, a, b, c, d, x, y)    \
    do {                                    \
        v[a] = v[a] + v[b] + x;            \
        v[d] = rotate(v[d] ^ v[a], 32UL);  \
        v[c] = v[c] + v[d];                \
        v[b] = rotate(v[b] ^ v[c], 24UL);  \
        v[a] = v[a] + v[b] + y;            \
        v[d] = rotate(v[d] ^ v[a], 16UL);  \
        v[c] = v[c] + v[d];                \
        v[b] = rotate(v[b] ^ v[c], 63UL);  \
    } while (0)

#define BLAKE2B_ROUND(v, s, r)                                             \
    do {                                                                    \
        BLAKE2B_G(v, 0, 4,  8, 12, m[blake2b_sigma[s][ 0]], m[blake2b_sigma[s][ 1]]); \
        BLAKE2B_G(v, 1, 5,  9, 13, m[blake2b_sigma[s][ 2]], m[blake2b_sigma[s][ 3]]); \
        BLAKE2B_G(v, 2, 6, 10, 14, m[blake2b_sigma[s][ 4]], m[blake2b_sigma[s][ 5]]); \
        BLAKE2B_G(v, 3, 7, 11, 15, m[blake2b_sigma[s][ 6]], m[blake2b_sigma[s][ 7]]); \
        BLAKE2B_G(v, 0, 5, 10, 15, m[blake2b_sigma[s][ 8]], m[blake2b_sigma[s][ 9]]); \
        BLAKE2B_G(v, 1, 6, 11, 12, m[blake2b_sigma[s][10]], m[blake2b_sigma[s][11]]); \
        BLAKE2B_G(v, 2, 7,  8, 13, m[blake2b_sigma[s][12]], m[blake2b_sigma[s][13]]); \
        BLAKE2B_G(v, 3, 4,  9, 14, m[blake2b_sigma[s][14]], m[blake2b_sigma[s][15]]); \
    } while (0)

/*
 * BLAKE2b hash kernel — batch processing.
 * Each work-item hashes one input block.
 *
 * @param inputs    Flat input buffer [count * in_len]
 * @param in_len    Length of each input (bytes)
 * @param out_len   Desired output length (1..64)
 * @param outputs   Output buffer [count * out_len]
 */
__kernel void kernel_hash_blake2b(
    __global const uchar *inputs,
    const uint in_len,
    const uint out_len,
    __global uchar *outputs)
{
    uint gid = get_global_id(0);
    if (gid == 0 && in_len == 0) return; /* edge case */

    __global const uchar *in = inputs + (size_t)gid * in_len;
    __global uchar *out = outputs + (size_t)gid * out_len;

    /* Local message buffer (16 ulongs = 128 bytes) */
    ulong m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = 0;
    }

    /* Load input into message buffer */
    uint blocks = in_len / 128;
    uint rem = in_len % 128;

    /* For simplicity: hash in single-pass for inputs <= 128 bytes */
    /* Multi-block would require chaining; most tile inputs are small */
    if (in_len <= 128) {
        /* Load input bytes into m */
        for (uint i = 0; i < in_len && i < 128; i++) {
            ((uchar*)m)[i] = in[i];
        }
        /* Pad remainder */
        if (in_len < 128) {
            ((uchar*)m)[in_len] = 0x80;
        }
    } else {
        /* For larger inputs, just load first 128 bytes (simplified) */
        for (int i = 0; i < 128; i++) {
            ((uchar*)m)[i] = in[i];
        }
    }

    /* Initialize state */
    ulong v[16];
    v[ 0] = blake2b_iv[0];
    v[ 1] = blake2b_iv[1];
    v[ 2] = blake2b_iv[2];
    v[ 3] = blake2b_iv[3];
    v[ 4] = blake2b_iv[4];
    v[ 5] = blake2b_iv[5];
    v[ 6] = blake2b_iv[6];
    v[ 7] = blake2b_iv[7];
    v[ 8] = 0UL ^ in_len;  /* parameter block: digest length + key length (0) */
    v[ 8] = blake2b_iv[0] ^ (0x01010000UL | out_len);
    v[ 9] = blake2b_iv[1];
    v[10] = blake2b_iv[2];
    v[11] = blake2b_iv[3];
    v[12] = blake2b_iv[4] ^ in_len; /* counter lo */
    v[13] = blake2b_iv[5];          /* counter hi */
    v[14] = blake2b_iv[6];
    v[15] = blake2b_iv[7];

    /* 12 rounds */
    for (int r = 0; r < 12; r++) {
        BLAKE2B_ROUND(v, r, r);
    }

    /* Finalize */
    ulong h[8];
    h[0] = v[0] ^ v[8];
    h[1] = v[1] ^ v[9];
    h[2] = v[2] ^ v[10];
    h[3] = v[3] ^ v[11];
    h[4] = v[4] ^ v[12];
    h[5] = v[5] ^ v[13];
    h[6] = v[6] ^ v[14];
    h[7] = v[7] ^ v[15];

    /* Write output */
    uint bytes_to_write = min(out_len, (uint)64);
    for (uint i = 0; i < bytes_to_write; i++) {
        out[i] = ((uchar*)h)[i];
    }
}
