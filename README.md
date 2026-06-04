# tile-opencl

Portable GPU compute kernels for the tile field system, written in OpenCL 1.2.

Works on **any** OpenCL 1.2+ runtime: NVIDIA, AMD, Intel, ARM Mali, and pocl (CPU).

## What It Does

Four kernels for tile field operations:

| Kernel | File | Description |
|--------|------|-------------|
| Hash | `kernel_hash.cl` | BLAKE2b batch hashing with constant memory for IV/sigma tables |
| Embed | `kernel_embed.cl` | Position-aware embedding with sinusoidal encoding, unit-normalized output |
| Search | `kernel_search.cl` | Cosine similarity search with local-memory top-K selection |
| Evolve | `kernel_evolve.cl` | Score evolution with atomic CAS updates, learning rate, clamping |

## Building

Requires an OpenCL 1.2+ SDK (headers + runtime library).

```bash
# Install dependencies (Debian/Ubuntu)
sudo apt install opencl-headers ocl-icd-opencl-dev

# Or with pocl (CPU OpenCL, works on ARM64)
sudo apt install opencl-headers pocl-opencl-icd

# Build
make

# Run tests
make test

# Run benchmarks (1K, 10K, 100K vectors)
make bench
```

## Usage

```c
#include "tile_opencl.h"

tile_context_t *ctx;
tile_device_info_t info;

// Initialize — auto-detects best device (GPU > CPU)
tile_init(&ctx, &info);

// Upload vector database
tile_upload_db(ctx, vectors, count, dim);

// Search
tile_search_result_t results[TILE_MAX_RESULTS];
uint32_t n_results;
tile_search(ctx, query, dim, results, &n_results);

// Evolve scores
tile_evolve(ctx, results, n_results, 0.1f, 0.0f, 2.0f);

// Cleanup
tile_cleanup(ctx);
```

## Device Info

On init, reports:
- Device name, vendor, OpenCL version
- GPU vs CPU
- Compute units, global memory
- Max workgroup size, local memory

## ARM64 / pocl Support

Designed to work on ARM64 devices with pocl (portable CPU OpenCL runtime):

```bash
sudo apt install pocl-opencl-icd
```

No GPU required — falls back to CPU automatically.

## Architecture

- **OpenCL C 1.2** kernels — maximum portability
- **Constant memory** for BLAKE2b lookup tables (IV, sigma)
- **Local memory** for search top-K reduction and embedding accumulation
- **Atomic CAS** for lock-free score evolution
- **CL_MEM_READ_ONLY** for database vectors, **CL_MEM_READ_WRITE** for scores
- **Benchmark mode** compares OpenCL vs CPU at 1K / 10K / 100K vectors

## License

MIT
