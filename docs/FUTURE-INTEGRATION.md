# Future Integration: tile-opencl

## Current State
Portable GPU compute kernels for the tile field system, written in OpenCL 1.2. Works on ANY OpenCL runtime: NVIDIA, AMD, Intel, ARM Mali, and pocl (CPU). Four kernels: hash (BLAKE2b), embed (position-aware), search (cosine similarity with top-K), evolve (atomic CAS score updates).

## Integration Opportunities

### With broad GPU support for rooms
tile-opencl ensures that room GPU acceleration works on ANY hardware — not just NVIDIA. AMD GPUs, Intel integrated graphics, ARM Mali (mobile), and even CPU (pocl) all run the same OpenCL kernels. This broadens the fleet's GPU coverage beyond the NVIDIA-only tile-cuda.

### With room-as-codespace heterogeneous hardware
Codespaces run on various cloud hardware. Some have NVIDIA GPUs (tile-cuda), some have AMD GPUs (tile-opencl), some have no GPU (CPU fallback). tile-opencl ensures room computation is GPU-accelerated wherever possible, regardless of GPU vendor.

### With construct-core hardware portability
construct-core's promise is hardware-agnostic agents. tile-opencl extends this to GPU computation: the same room logic accelerates on any GPU via OpenCL. The agent doesn't know or care what GPU is underneath.

## Dormant Ideas Now Unlockable
OpenCL support was for "GPU coverage." Now it's essential for fleet-wide room deployment. The fleet can't be NVIDIA-only — that's a single-vendor dependency. tile-opencl provides vendor independence.

## Potential in Mature Systems
Every GPU in the fleet is utilized, regardless of vendor. tile-cuda for NVIDIA (fastest), tile-opencl for everything else (portable), tile-neon for ARM (edge). The fleet extracts maximum compute from all available hardware.

## Cross-Pollination Ideas
- **tile-cuda**: NVIDIA-specific counterpart (faster but less portable)
- **tile-neon**: ARM counterpart for edge
- **forgemaster**: Orchestrates which kernel set to use per GPU

## Dependencies for Next Steps
- Testing on AMD and Intel GPUs
- Integration with Forgemaster's GPU dispatch
- Performance benchmarking vs tile-cuda
