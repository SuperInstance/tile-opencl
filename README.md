# Tile OpenCL

**Tile OpenCL** provides portable GPU acceleration for ternary neural network kernels using OpenCL — the vendor-neutral compute framework that runs on NVIDIA, AMD, Intel, and mobile GPUs. It includes OpenCL kernels for hashing, embedding, vector search, and evolutionary mutation, accessible from a C host API.

## Why It Matters

CUDA is NVIDIA-only. NEON is ARM-only. OpenCL is the only compute framework that runs on *everything*: NVIDIA GPUs, AMD GPUs, Intel integrated graphics, mobile Adreno/Mali GPUs, and even FPGAs. For the SuperInstance ecosystem, this means ternary kernels can run on whatever hardware the fleet has — a Raspberry Pi with integrated graphics, a cloud server with an AMD GPU, or an Intel NUC with Iris graphics. Tile OpenCL provides the same core operations as Tile CUDA (hash, embed, search, evolve) but in portable OpenCL C, making ternary computation hardware-agnostic.

## How It Works

### OpenCL Execution Model

OpenCL follows a hierarchical execution model:

```
Host (CPU) → Command Queue → Kernel → NDRange
                                      ├── Work-groups (compute units)
                                      │   └── Work-items (threads)
                                      └── Global ID space
```

The host (`tile_opencl.c`) manages context, queues, and buffers. Kernels (`.cl` files) execute on the device.

### Kernel Suite

| File | Operation | Optimization |
|------|-----------|-------------|
| `kernel_hash.cl` | BLAKE2b hashing | Vectorized XOR/shift, 4 work-items per hash |
| `kernel_embed.cl` | Embedding lookup | Coalesced memory access, constant cache |
| `kernel_search.cl` | Top-K cosine similarity | Local memory reduction, wavefront shuffle |
| `kernel_evolve.cl` | Genetic mutation | PRNG per work-item, branchless ternary mutation |

### Ternary Search Kernel

The top-K search kernel computes cosine similarity between a query and all database vectors:

```opencl
__kernel void search_topk(
    __global const char *db,     // database: N × dim ternary values
    __global const char *query,  // dim ternary values
    __global float *scores,      // N similarity scores
    int n, int dim)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float dot = 0;
    for (int j = 0; j < dim; j++) {
        char a = db[i * dim + j];
        char b = query[j];
        if (a != 0 && b != 0)
            dot += (a == b) ? 1.0f : -1.0f;
    }
    scores[i] = dot / (float)dim;  // normalized cosine
}
```

### Memory Model

OpenCL defines four memory regions:

| Region | Scope | Bandwidth |
|--------|-------|-----------|
| Global | All work-items | ~1 TB/s (device-dependent) |
| Constant | All work-items (read-only) | Cached, ~4 TB/s |
| Local | Work-group only | ~10 TB/s |
| Private | Work-item only | Register speed |

Kernels use `__local` memory for group-wide reductions and `__global` for input/output.

### Complexity

Identical to CUDA equivalents: O(N × D) for batch operations. The constant factor depends on the device — discrete GPUs are ~10× faster than integrated.

## Quick Start

```c
#include <CL/cl.h>
#include "tile_opencl.h"

int main(void) {
    TileContext ctx;
    tile_init(&ctx);  // detect platform, create context+queue
    tile_load_kernels(&ctx, "src/kernel_search.cl");

    // Run vector search
    int n = 10000, dim = 384;
    char *db = /* ternary database */;
    char *query = /* query vector */;
    float *scores = malloc(n * sizeof(float));

    tile_search(&ctx, db, query, scores, n, dim);

    // Top-5 results
    for (int i = 0; i < 5; i++)
        printf("Result %d: score = %.4f\n", i, scores[i]);

    tile_cleanup(&ctx);
    return 0;
}
```

Build: `gcc -lOpenCL -o tile_opencl src/tile_opencl.c -I/usr/include/CL`

## API

| Function | Description |
|----------|-------------|
| `tile_init(ctx)` | Detect platform, device, create context |
| `tile_load_kernels(ctx, path)` | Compile .cl kernels |
| `tile_hash(ctx, input, output, n)` | Parallel hashing |
| `tile_embed(ctx, ids, weights, output, n)` | Embedding lookup |
| `tile_search(ctx, db, query, scores, n, dim)` | Vector similarity search |
| `tile_evolve(ctx, population, n, len, rate)` | Genetic mutation |
| `tile_cleanup(ctx)` | Release resources |

## Architecture Notes

Tile OpenCL provides the same γ (constructive) computation as Tile CUDA and Tile NEON, but on any device. This is critical for fleet diversity — agents running on AMD GPUs, Intel iGPUs, or mobile GPUs all get hardware-accelerated ternary operations. In the γ + η = C framework, OpenCL ensures that γ is not hardware-gated: the constructive computation substrate is available wherever the fleet operates. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Khronos Group. (2024). *OpenCL Specification, Version 3.0*. — The OpenCL standard.
2. Gaster, B. R., et al. (2012). *Heterogeneous Computing with OpenCL*, 2nd ed. Morgan Kaufmann.
3. Stone, J. E., Gohara, D., & Shi, G. (2010). "OpenCL: A Parallel Programming Standard for Heterogeneous Computing Systems." *Computing in Science & Engineering*, 12(3), 66–73.

## License

MIT
