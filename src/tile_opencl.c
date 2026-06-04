/*
 * tile_opencl.c — Host code: device detection, buffer mgmt, kernel launch
 * OpenCL 1.2 compatible
 */
#define _POSIX_C_SOURCE 199309L
#define min(a,b) ((a)<(b)?(a):(b))

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "tile_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Internal context */
struct tile_context {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       ctx;
    cl_command_queue queue;
    cl_program       program;

    /* Kernels */
    cl_kernel k_hash;
    cl_kernel k_embed;
    cl_kernel k_search;
    cl_kernel k_search_reduce;
    cl_kernel k_evolve;
    cl_kernel k_evolve_decay;

    /* DB buffers */
    cl_mem db_vectors;
    uint32_t db_count;
    uint32_t db_dim;
    int db_loaded;

    /* Score buffer */
    cl_mem scores;
    uint32_t scores_count;
};

/* ---- Error handling ---- */
static const char *cl_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        default: return "UNKNOWN";
    }
}

#define CL_CHECK(expr) do { \
    cl_int _err = (expr); \
    if (_err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error at %s:%d: %s (0x%d)\n", \
                __FILE__, __LINE__, cl_err_str(_err), _err); \
        return -1; \
    } \
} while (0)

/* ---- Kernel source loading ---- */
static char *load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open kernel file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* ---- Device selection ---- */
static int select_device(tile_context_t *tc, tile_device_info_t *info) {
    cl_uint num_platforms;
    CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return -1;
    }

    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    int best_gpu = -1;
    int best_cpu = -1;
    cl_device_id best_gpu_dev = NULL;
    cl_device_id best_cpu_dev = NULL;
    cl_platform_id best_gpu_plat = NULL;
    cl_platform_id best_cpu_plat = NULL;

    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        cl_device_id *devices = malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint d = 0; d < num_devices; d++) {
            cl_device_type dtype;
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);

            cl_uint cu;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);

            if (dtype & CL_DEVICE_TYPE_GPU) {
                if ((int)cu > best_gpu) {
                    best_gpu = cu;
                    best_gpu_dev = devices[d];
                    best_gpu_plat = platforms[p];
                }
            } else {
                if ((int)cu > best_cpu) {
                    best_cpu = cu;
                    best_cpu_dev = devices[d];
                    best_cpu_plat = platforms[p];
                }
            }
        }
        free(devices);
    }
    free(platforms);

    /* Prefer GPU, fallback to CPU */
    if (best_gpu_dev) {
        tc->device = best_gpu_dev;
        tc->platform = best_gpu_plat;
        info->is_gpu = 1;
    } else if (best_cpu_dev) {
        tc->device = best_cpu_dev;
        tc->platform = best_cpu_plat;
        info->is_gpu = 0;
    } else {
        fprintf(stderr, "No OpenCL devices found\n");
        return -1;
    }

    /* Fill device info */
    clGetDeviceInfo(tc->device, CL_DEVICE_NAME, sizeof(info->name), info->name, NULL);
    clGetDeviceInfo(tc->device, CL_DEVICE_VENDOR, sizeof(info->vendor), info->vendor, NULL);
    clGetDeviceInfo(tc->device, CL_DEVICE_VERSION, sizeof(info->version), info->version, NULL);
    cl_uint cu;
    clGetDeviceInfo(tc->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    info->compute_units = cu;
    clGetDeviceInfo(tc->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info->global_mem), &info->global_mem, NULL);
    clGetDeviceInfo(tc->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info->max_workgroup_size), &info->max_workgroup_size, NULL);
    cl_ulong lmem;
    clGetDeviceInfo(tc->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL);
    info->local_mem_size = lmem;

    return 0;
}

/* ---- Public API ---- */

int tile_init(tile_context_t **out_ctx, tile_device_info_t *out_info) {
    tile_context_t *tc = calloc(1, sizeof(tile_context_t));
    if (!tc) return -1;

    memset(out_info, 0, sizeof(*out_info));

    if (select_device(tc, out_info) != 0) {
        free(tc);
        return -1;
    }

    printf("[tile-opencl] Device: %s\n", out_info->name);
    printf("[tile-opencl] Vendor: %s\n", out_info->vendor);
    printf("[tile-opencl] Version: %s\n", out_info->version);
    printf("[tile-opencl] Type: %s\n", out_info->is_gpu ? "GPU" : "CPU");
    printf("[tile-opencl] Compute units: %u\n", out_info->compute_units);
    printf("[tile-opencl] Global memory: %lu MB\n", (unsigned long)(out_info->global_mem / (1024*1024)));
    printf("[tile-opencl] Max workgroup size: %zu\n", out_info->max_workgroup_size);
    printf("[tile-opencl] Local memory: %lu KB\n", (unsigned long)(out_info->local_mem_size / 1024));

    /* Create context */
    cl_int err;
    tc->ctx = clCreateContext(NULL, 1, &tc->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context: %s\n", cl_err_str(err));
        free(tc);
        return -1;
    }

    /* Create command queue */
    tc->queue = clCreateCommandQueue(tc->ctx, tc->device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue: %s\n", cl_err_str(err));
        clReleaseContext(tc->ctx);
        free(tc);
        return -1;
    }

    /* Load and build kernels */
    const char *kernel_files[] = {
        "src/kernel_hash.cl",
        "src/kernel_embed.cl",
        "src/kernel_search.cl",
        "src/kernel_evolve.cl"
    };
    const int n_kernels = 4;

    char *sources[4];
    const char *source_ptrs[4];
    size_t source_sizes[4];

    for (int i = 0; i < n_kernels; i++) {
        sources[i] = load_file(kernel_files[i]);
        if (!sources[i]) {
            fprintf(stderr, "Failed to load kernel: %s\n", kernel_files[i]);
            for (int j = 0; j < i; j++) free(sources[j]);
            clReleaseCommandQueue(tc->queue);
            clReleaseContext(tc->ctx);
            free(tc);
            return -1;
        }
        source_ptrs[i] = sources[i];
        source_sizes[i] = strlen(sources[i]);
    }

    tc->program = clCreateProgramWithSource(tc->ctx, n_kernels, source_ptrs, source_sizes, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program: %s\n", cl_err_str(err));
        for (int i = 0; i < n_kernels; i++) free(sources[i]);
        clReleaseCommandQueue(tc->queue);
        clReleaseContext(tc->ctx);
        free(tc);
        return -1;
    }

    /* Build with OpenCL 1.2 target */
    err = clBuildProgram(tc->program, 1, &tc->device,
                         "-cl-std=CL1.2 -Werror", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(tc->program, tc->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = malloc(log_size + 1);
        clGetProgramBuildInfo(tc->program, tc->device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Kernel build failed:\n%s\n", log);
        free(log);
        clReleaseProgram(tc->program);
        clReleaseCommandQueue(tc->queue);
        clReleaseContext(tc->ctx);
        free(tc);
        return -1;
    }

    for (int i = 0; i < n_kernels; i++) free(sources[i]);
    printf("[tile-opencl] Kernels built successfully\n");

    /* Create kernels */
    tc->k_hash = clCreateKernel(tc->program, "kernel_hash_blake2b", &err);
    tc->k_embed = clCreateKernel(tc->program, "kernel_embed", &err);
    tc->k_search = clCreateKernel(tc->program, "kernel_search", &err);
    tc->k_search_reduce = clCreateKernel(tc->program, "kernel_search_reduce", &err);
    tc->k_evolve = clCreateKernel(tc->program, "kernel_evolve", &err);
    tc->k_evolve_decay = clCreateKernel(tc->program, "kernel_evolve_decay", &err);

    *out_ctx = tc;
    return 0;
}

void tile_cleanup(tile_context_t *tc) {
    if (!tc) return;
    if (tc->db_vectors) clReleaseMemObject(tc->db_vectors);
    if (tc->scores) clReleaseMemObject(tc->scores);
    clReleaseKernel(tc->k_hash);
    clReleaseKernel(tc->k_embed);
    clReleaseKernel(tc->k_search);
    clReleaseKernel(tc->k_search_reduce);
    clReleaseKernel(tc->k_evolve);
    clReleaseKernel(tc->k_evolve_decay);
    clReleaseProgram(tc->program);
    clReleaseCommandQueue(tc->queue);
    clReleaseContext(tc->ctx);
    free(tc);
}

int tile_upload_db(tile_context_t *tc, const float *vectors,
                   uint32_t count, uint32_t dim) {
    if (!tc || !vectors || count == 0) return -1;

    if (tc->db_vectors) clReleaseMemObject(tc->db_vectors);

    size_t bytes = (size_t)count * dim * sizeof(float);
    cl_int err;
    tc->db_vectors = clCreateBuffer(tc->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     bytes, (void*)vectors, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create DB buffer: %s\n", cl_err_str(err));
        return -1;
    }

    tc->db_count = count;
    tc->db_dim = dim;
    tc->db_loaded = 1;

    /* Create score buffer */
    if (tc->scores) clReleaseMemObject(tc->scores);
    tc->scores = clCreateBuffer(tc->ctx, CL_MEM_READ_WRITE,
                                count * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create score buffer: %s\n", cl_err_str(err));
        return -1;
    }
    tc->scores_count = count;

    /* Initialize scores to 0.5 */
    float *init = malloc(count * sizeof(float));
    for (uint32_t i = 0; i < count; i++) init[i] = 0.5f;
    clEnqueueWriteBuffer(tc->queue, tc->scores, CL_TRUE, 0,
                         count * sizeof(float), init, 0, NULL, NULL);
    free(init);

    printf("[tile-opencl] Uploaded DB: %u vectors x %u dims (%zu KB)\n",
           count, dim, bytes / 1024);
    return 0;
}

int tile_hash_blake2b(tile_context_t *tc,
                      const uint8_t *input, size_t in_len, uint32_t count,
                      uint8_t *output) {
    if (!tc || !input || !output) return -1;

    cl_int err;
    size_t total_in = (size_t)count * in_len;
    size_t total_out = (size_t)count * TILE_BLAKE2B_OUTLEN;

    cl_mem d_in = clCreateBuffer(tc->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  total_in, (void*)input, &err);
    if (err != CL_SUCCESS) return -1;

    cl_mem d_out = clCreateBuffer(tc->ctx, CL_MEM_WRITE_ONLY,
                                   total_out, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_in);
        return -1;
    }

    uint32_t out_len = TILE_BLAKE2B_OUTLEN;
    clSetKernelArg(tc->k_hash, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(tc->k_hash, 1, sizeof(uint32_t), &in_len);
    clSetKernelArg(tc->k_hash, 2, sizeof(uint32_t), &out_len);
    clSetKernelArg(tc->k_hash, 3, sizeof(cl_mem), &d_out);

    size_t global = count;
    err = clEnqueueNDRangeKernel(tc->queue, tc->k_hash, 1, NULL,
                                  &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Hash kernel launch failed: %s\n", cl_err_str(err));
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        return -1;
    }

    clEnqueueReadBuffer(tc->queue, d_out, CL_TRUE, 0, total_out, output, 0, NULL, NULL);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    return 0;
}

int tile_embed(tile_context_t *tc,
               const uint8_t *hash_input, size_t in_len, uint32_t count,
               float *embeddings, uint32_t dim) {
    if (!tc || !hash_input || !embeddings) return -1;

    cl_int err;
    size_t total_in = (size_t)count * in_len;
    size_t total_out = (size_t)count * dim * sizeof(float);

    cl_mem d_in = clCreateBuffer(tc->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  total_in, (void*)hash_input, &err);
    if (err != CL_SUCCESS) return -1;

    cl_mem d_out = clCreateBuffer(tc->ctx, CL_MEM_WRITE_ONLY,
                                   total_out, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_in);
        return -1;
    }

    clSetKernelArg(tc->k_embed, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(tc->k_embed, 1, sizeof(uint32_t), &in_len);
    clSetKernelArg(tc->k_embed, 2, sizeof(uint32_t), &count);
    clSetKernelArg(tc->k_embed, 3, sizeof(cl_mem), &d_out);
    clSetKernelArg(tc->k_embed, 4, sizeof(uint32_t), &dim);

    size_t global = count;
    err = clEnqueueNDRangeKernel(tc->queue, tc->k_embed, 1, NULL,
                                  &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        return -1;
    }

    clEnqueueReadBuffer(tc->queue, d_out, CL_TRUE, 0, total_out, embeddings, 0, NULL, NULL);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    return 0;
}

int tile_search(tile_context_t *tc,
                const float *query, uint32_t dim,
                tile_search_result_t *results, uint32_t *n_results) {
    if (!tc || !tc->db_loaded || !query || !results) return -1;

    cl_int err;
    cl_mem d_query = clCreateBuffer(tc->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     dim * sizeof(float), (void*)query, &err);
    if (err != CL_SUCCESS) return -1;

    /* Intermediate results: one set per work-group */
    size_t lws = 256;
    size_t gws = ((tc->db_count + lws - 1) / lws) * lws;
    uint32_t num_groups = (uint32_t)(gws / lws);
    size_t group_results_size = (size_t)num_groups * TILE_MAX_RESULTS * sizeof(tile_search_result_t);

    cl_mem d_group_results = clCreateBuffer(tc->ctx, CL_MEM_READ_WRITE,
                                             group_results_size, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query);
        return -1;
    }

    cl_mem d_final = clCreateBuffer(tc->ctx, CL_MEM_WRITE_ONLY,
                                     TILE_MAX_RESULTS * sizeof(tile_search_result_t), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query);
        clReleaseMemObject(d_group_results);
        return -1;
    }

    /* Launch search kernel */
    clSetKernelArg(tc->k_search, 0, sizeof(cl_mem), &tc->db_vectors);
    clSetKernelArg(tc->k_search, 1, sizeof(cl_mem), &d_query);
    clSetKernelArg(tc->k_search, 2, sizeof(uint32_t), &tc->db_count);
    clSetKernelArg(tc->k_search, 3, sizeof(uint32_t), &dim);
    clSetKernelArg(tc->k_search, 4, sizeof(cl_mem), &d_group_results);

    err = clEnqueueNDRangeKernel(tc->queue, tc->k_search, 1, NULL,
                                  &gws, &lws, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Search kernel launch failed: %s\n", cl_err_str(err));
        clReleaseMemObject(d_query);
        clReleaseMemObject(d_group_results);
        clReleaseMemObject(d_final);
        return -1;
    }

    /* Launch reduction kernel */
    clSetKernelArg(tc->k_search_reduce, 0, sizeof(cl_mem), &d_group_results);
    clSetKernelArg(tc->k_search_reduce, 1, sizeof(uint32_t), &num_groups);
    clSetKernelArg(tc->k_search_reduce, 2, sizeof(cl_mem), &d_final);

    size_t reduce_lws = (lws < (size_t)TILE_MAX_RESULTS) ? lws : (size_t)TILE_MAX_RESULTS;
    size_t reduce_gws = reduce_lws;
    err = clEnqueueNDRangeKernel(tc->queue, tc->k_search_reduce, 1, NULL,
                                  &reduce_gws, &reduce_lws, 0, NULL, NULL);

    /* Read results */
    uint8_t *raw = malloc(TILE_MAX_RESULTS * sizeof(tile_search_result_t));
    clEnqueueReadBuffer(tc->queue, d_final, CL_TRUE, 0,
                        TILE_MAX_RESULTS * sizeof(tile_search_result_t), raw, 0, NULL, NULL);

    /* Parse results (struct layout: {uint32_t index, float score} = 8 bytes) */
    *n_results = 0;
    for (uint32_t i = 0; i < TILE_MAX_RESULTS; i++) {
        uint32_t idx;
        float score;
        memcpy(&idx, raw + i * 8, 4);
        memcpy(&score, raw + i * 8 + 4, 4);
        if (score > -1.5f) {
            results[*n_results].index = idx;
            results[*n_results].score = score;
            (*n_results)++;
        }
    }

    /* Sort by score descending */
    for (uint32_t i = 0; i < *n_results; i++) {
        for (uint32_t j = i + 1; j < *n_results; j++) {
            if (results[j].score > results[i].score) {
                tile_search_result_t tmp = results[i];
                results[i] = results[j];
                results[j] = tmp;
            }
        }
    }

    free(raw);
    clReleaseMemObject(d_query);
    clReleaseMemObject(d_group_results);
    clReleaseMemObject(d_final);
    return 0;
}

int tile_evolve(tile_context_t *tc,
                const tile_search_result_t *results, uint32_t n_results,
                float lr, float clamp_min, float clamp_max) {
    if (!tc || !results || n_results == 0) return -1;

    cl_int err;
    uint32_t *indices = malloc(n_results * sizeof(uint32_t));
    float *sims = malloc(n_results * sizeof(float));
    for (uint32_t i = 0; i < n_results; i++) {
        indices[i] = results[i].index;
        sims[i] = results[i].score;
    }

    cl_mem d_indices = clCreateBuffer(tc->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       n_results * sizeof(uint32_t), indices, &err);
    cl_mem d_sims = clCreateBuffer(tc->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n_results * sizeof(float), sims, &err);
    free(indices);
    free(sims);
    if (err != CL_SUCCESS) return -1;

    clSetKernelArg(tc->k_evolve, 0, sizeof(cl_mem), &tc->scores);
    clSetKernelArg(tc->k_evolve, 1, sizeof(cl_mem), &d_indices);
    clSetKernelArg(tc->k_evolve, 2, sizeof(cl_mem), &d_sims);
    clSetKernelArg(tc->k_evolve, 3, sizeof(uint32_t), &n_results);
    clSetKernelArg(tc->k_evolve, 4, sizeof(float), &lr);
    clSetKernelArg(tc->k_evolve, 5, sizeof(float), &clamp_min);
    clSetKernelArg(tc->k_evolve, 6, sizeof(float), &clamp_max);
    clSetKernelArg(tc->k_evolve, 7, sizeof(uint32_t), &tc->scores_count);

    size_t global = n_results;
    err = clEnqueueNDRangeKernel(tc->queue, tc->k_evolve, 1, NULL,
                                  &global, NULL, 0, NULL, NULL);

    clReleaseMemObject(d_indices);
    clReleaseMemObject(d_sims);
    return (err == CL_SUCCESS) ? 0 : -1;
}

/* ---- CPU reference implementations ---- */
static void cpu_cosine_sim(const float *db, const float *query,
                           uint32_t count, uint32_t dim,
                           tile_search_result_t *results, uint32_t *n_results) {
    float *scores = malloc(count * sizeof(float));
    for (uint32_t i = 0; i < count; i++) {
        float dot = 0, norm_q = 0, norm_v = 0;
        const float *v = db + (size_t)i * dim;
        for (uint32_t d = 0; d < dim; d++) {
            dot += v[d] * query[d];
            norm_q += query[d] * query[d];
            norm_v += v[d] * v[d];
        }
        float denom = sqrtf(norm_q) * sqrtf(norm_v);
        scores[i] = (denom > 1e-6f) ? dot / denom : 0.0f;
    }

    /* Find top-K */
    *n_results = (count < TILE_MAX_RESULTS) ? count : TILE_MAX_RESULTS;
    for (uint32_t i = 0; i < *n_results; i++) {
        results[i].score = -2.0f;
        results[i].index = 0;
    }

    for (uint32_t i = 0; i < count; i++) {
        float s = scores[i];
        /* Find worst in current top-K */
        int worst = 0;
        for (uint32_t k = 1; k < *n_results; k++) {
            if (results[k].score < results[worst].score) worst = k;
        }
        if (s > results[worst].score) {
            results[worst].score = s;
            results[worst].index = i;
        }
    }

    /* Sort descending */
    for (uint32_t i = 0; i < *n_results; i++) {
        for (uint32_t j = i + 1; j < *n_results; j++) {
            if (results[j].score > results[i].score) {
                tile_search_result_t tmp = results[i];
                results[i] = results[j];
                results[j] = tmp;
            }
        }
    }
    free(scores);
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int tile_read_scores(tile_context_t *tc, float *scores, uint32_t count) {
    if (!tc || !scores) return -1;
    cl_int err = clEnqueueReadBuffer(tc->queue, tc->scores, CL_TRUE, 0,
                                     count * sizeof(float), scores, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

int tile_benchmark(tile_context_t *tc, uint32_t count, uint32_t dim) {
    printf("\n=== Benchmark: %u vectors x %u dims ===\n", count, dim);

    /* Generate random test data */
    float *db = malloc((size_t)count * dim * sizeof(float));
    float *query = malloc(dim * sizeof(float));

    srand(42);
    for (size_t i = 0; i < (size_t)count * dim; i++) {
        db[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    /* Normalize DB vectors */
    for (uint32_t i = 0; i < count; i++) {
        float norm = 0;
        for (uint32_t d = 0; d < dim; d++) {
            float v = db[(size_t)i * dim + d];
            norm += v * v;
        }
        norm = sqrtf(norm);
        if (norm > 1e-6f) {
            for (uint32_t d = 0; d < dim; d++) {
                db[(size_t)i * dim + d] /= norm;
            }
        }
    }

    for (uint32_t d = 0; d < dim; d++) {
        query[d] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    float qnorm = 0;
    for (uint32_t d = 0; d < dim; d++) qnorm += query[d] * query[d];
    qnorm = sqrtf(qnorm);
    for (uint32_t d = 0; d < dim; d++) query[d] /= qnorm;

    /* Upload to device */
    if (tile_upload_db(tc, db, count, dim) != 0) {
        free(db); free(query);
        return -1;
    }

    /* CPU search */
    tile_search_result_t cpu_results[TILE_MAX_RESULTS];
    uint32_t cpu_n;
    double t0 = get_time_ms();
    cpu_cosine_sim(db, query, count, dim, cpu_results, &cpu_n);
    double cpu_ms = get_time_ms() - t0;

    /* OpenCL search */
    tile_search_result_t gpu_results[TILE_MAX_RESULTS];
    uint32_t gpu_n;
    t0 = get_time_ms();
    int ret = tile_search(tc, query, dim, gpu_results, &gpu_n);
    double gpu_ms = get_time_ms() - t0;

    printf("  CPU search:  %.2f ms\n", cpu_ms);
    printf("  OpenCL search: %.2f ms\n", gpu_ms);
    if (gpu_ms > 0.01 && cpu_ms > 0.01) {
        printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);
    }

    /* Compare top results */
    printf("  CPU top-3: ");
    for (uint32_t i = 0; i < (cpu_n < 3 ? cpu_n : 3); i++)
        printf("[%u: %.4f] ", cpu_results[i].index, cpu_results[i].score);
    printf("\n  GPU top-3: ");
    for (uint32_t i = 0; i < (gpu_n < 3 ? gpu_n : 3); i++)
        printf("[%u: %.4f] ", gpu_results[i].index, gpu_results[i].score);
    printf("\n");

    free(db);
    free(query);
    return ret;
}
