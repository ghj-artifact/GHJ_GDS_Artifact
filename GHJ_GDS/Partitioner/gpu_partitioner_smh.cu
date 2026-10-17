/*
gpu_partitioner_smh.cu - partitioner using a shared-memory atomic histogram

Build: compile as a translation unit of the main program.
Use only through GPUPartitioner::preprocess_gpu_gds(bool isR, const int* data, size_t n).

Implementation:
  - Partitioning kernel: shared-memory atomic histogram in
    compute_partitions_and_offsets_kernel.
  - Spill pipeline: ping-pong cuFile writes with two slots, background write threads,
    and parallel file registration. Input is read from pinned host memory.
  - Partition weights are represented by repeated base_paths entries.

Only the partitioning kernel differs from the WPS implementation. The output
semantics are unchanged: d_local_offsets[idx] remains in [0, count[p]-1].
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "gpu_partitioner_smh.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <cufile.h>

// Macros.

#define checkCuda(err) __checkCuda((err), __FILE__, __LINE__)
static inline void __checkCuda(cudaError_t r, const char* f, int l) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s  at %s:%d\n", cudaGetErrorString(r), f, l);
        assert(r == cudaSuccess);
    }
}

// Constants.

static constexpr size_t ALIGNMENT      = 4096;
static constexpr size_t TARGET_BATCHES = 6;

static inline size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// Global timing counters.

static std::mutex  g_stats_mutex;
static double g_total_h2d_ms     = 0.0;
static double g_total_compute_ms = 0.0;
static double g_total_scatter_ms = 0.0;
static double g_total_wait_ms    = 0.0;
static int    g_total_batches    = 0;

// ── GPU Kernels ───────────────────────────────────────────────────────────────

// Shared-memory atomic histogram: compute partition IDs and local offsets.
__global__ void compute_partitions_and_offsets_kernel(
    const int* data,
    int* partitions,
    int* partition_offsets,
    uint64_t n,
    int partition_count,
    int* d_local_offsets)
{
    extern __shared__ int shared_histogram[];

    uint64_t idx       = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int      thread_id = threadIdx.x;

    // Phase 1: initialize the shared histogram.
    for (int i = thread_id; i < partition_count; i += blockDim.x)
        shared_histogram[i] = 0;
    __syncthreads();

    // Phase 1 continued: compute partition IDs and update the shared histogram.
    if (idx < n) {
        unsigned int hash = (unsigned int)data[idx];
        hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
        hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
        hash = (hash >> 16) ^ hash;
        int partition_id = (int)((unsigned int)hash % (unsigned int)partition_count);
        partitions[idx] = partition_id;
        atomicAdd(&shared_histogram[partition_id], 1);
    }
    __syncthreads();

    // Phase 2: reserve this block's per-partition counts in global partition_offsets
    // and replace shared_histogram[i] with this block's global start offset.
    for (int i = thread_id; i < partition_count; i += blockDim.x) {
        int local_count = shared_histogram[i];
        if (local_count > 0) {
            int global_offset = atomicAdd(&partition_offsets[i], local_count);
            shared_histogram[i] = global_offset;
        }
    }
    __syncthreads();

    // Phase 3: assign each element a unique offset within its partition.
    if (idx < n) {
        int partition_id = partitions[idx];
        int local_offset = atomicAdd(&shared_histogram[partition_id], 1);
        d_local_offsets[idx] = local_offset;
    }
}

// Exclusive prefix sum -> d_partition_starts (scatter start offsets).
__global__ void exclusive_prefix_sum(const int* counts, int* starts, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int acc = 0;
        for (int i = 0; i < n; ++i) { starts[i] = acc; acc += counts[i]; }
    }
}

// Convert local offsets to global flat offsets by adding each partition's start.
__global__ void add_base_offsets(
    int* d_local_offsets, const int* d_partition_starts,
    const int* d_partitions, uint64_t n)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_local_offsets[idx] += d_partition_starts[d_partitions[idx]];
}

// Flat scatter: write each element to its global offset in the scatter buffer.
__global__ void flat_scatter(
    const int* data, const int* d_flat_offsets,
    int* d_scatter, uint64_t n)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_scatter[d_flat_offsets[idx]] = data[idx];
}

// Constructor and destructor.

GPUPartitioner::GPUPartitioner(int partition_count, int device_id,
                               const std::vector<std::string>& base_paths,
                               size_t max_batch_size)
    : partition_count_(partition_count), device_id_(device_id),
      items_per_thread_(1), max_batch_size_(max_batch_size),
      base_paths_(base_paths),
      partition_files_(partition_count),
      partition_fds_(partition_count, -1),
      partition_handles_(partition_count, nullptr),
      file_elem_offsets_(partition_count, 0)
{
    auto t0 = std::chrono::steady_clock::now();
    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS)
        fprintf(stderr, "[smh] cuFileDriverOpen failed: %d\n", (int)st.err);
    double driver_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    auto t1 = std::chrono::steady_clock::now();
    checkCuda(cudaSetDevice(device_id_));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device_id_));
    double cuda_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();

    std::cout << "[smh] Device " << device_id_ << ": " << prop.name << "\n";
    std::cout << "[smh] partition_count=" << partition_count_
              << "  storage_dirs=" << base_paths_.size() << "\n";
    std::cout << "[smh] ctor: cuFileDriverOpen=" << std::fixed << std::setprecision(1)
              << driver_ms << "ms  cudaSetDevice+Props=" << cuda_ms << "ms\n";
    std::cout << "[smh] compute kernel: shared memory atomic histogram\n";
}

GPUPartitioner::~GPUPartitioner() {
    cleanup_persistent_resources();
    cleanup_file_resources();
    cuFileDriverClose();
}

void GPUPartitioner::set_items_per_thread(int val) {
    (void)val;  // The histogram kernel always uses one element per thread.
}

// cleanup_file_resources

void GPUPartitioner::cleanup_file_resources() {
    for (int p = 0; p < partition_count_; ++p) {
        if (partition_handles_[p]) {
            cuFileHandleDeregister(partition_handles_[p]);
            partition_handles_[p] = nullptr;
        }
        if (partition_fds_[p] >= 0) {
            close(partition_fds_[p]);
            partition_fds_[p] = -1;
        }
    }
    std::fill(file_elem_offsets_.begin(), file_elem_offsets_.end(), 0LL);
}

// cleanup_persistent_resources

void GPUPartitioner::cleanup_persistent_resources() {
    for (int s = 0; s < NUM_SLOTS; ++s)
        wait_slot_writes(s);

    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }

    for (int s = 0; s < NUM_SLOTS; ++s) {
        if (d_scatter_[s]) {
            cuFileBufDeregister(d_scatter_[s]);
            cudaFree(d_scatter_[s]);
            d_scatter_[s] = nullptr;
        }
    }
    scatter_reg_size_ = 0;

    for (int s = 0; s < NUM_SLOTS; ++s) {
        if (h_pinned_part_sizes_[s]) {
            cudaFreeHost(h_pinned_part_sizes_[s]);
            h_pinned_part_sizes_[s] = nullptr;
        }
        if (h_pinned_part_starts_[s]) {
            cudaFreeHost(h_pinned_part_starts_[s]);
            h_pinned_part_starts_[s] = nullptr;
        }
    }

    if (d_data_pool_)          { cudaFree(d_data_pool_);          d_data_pool_          = nullptr; }
    if (d_partitions_pool_)    { cudaFree(d_partitions_pool_);    d_partitions_pool_    = nullptr; }
    if (d_local_offsets_pool_) { cudaFree(d_local_offsets_pool_); d_local_offsets_pool_ = nullptr; }
    if (d_part_offsets_pool_)  { cudaFree(d_part_offsets_pool_);  d_part_offsets_pool_  = nullptr; }
    if (d_part_starts_pool_)   { cudaFree(d_part_starts_pool_);   d_part_starts_pool_   = nullptr; }
    pool_max_batch_size_ = 0;

    cleanup_file_resources();
}

// calculate_adaptive_batch_size

size_t GPUPartitioner::calculate_adaptive_batch_size(size_t remaining_elements) {
    size_t free_mem, total_mem;
    checkCuda(cudaMemGetInfo(&free_mem, &total_mem));
    size_t per_elem = (5 + (size_t)NUM_SLOTS) * sizeof(int);
    size_t max_by_mem = (size_t)(free_mem * 0.70) / per_elem;
    size_t batch_size = std::min({max_by_mem, remaining_elements, max_batch_size_});
    return std::max(batch_size, (size_t)1);
}

// wait_slot_writes

void GPUPartitioner::wait_slot_writes(int slot) {
    for (auto& t : slot_threads_[slot])
        if (t.joinable()) t.join();
    slot_threads_[slot].clear();

    for (const auto& r : slot_write_results_[slot]) {
        if (r.written != (ssize_t)r.requested)
            fprintf(stderr, "[smh] Incomplete write: requested=%zu written=%zd\n",
                    r.requested, r.written);
    }
    slot_write_results_[slot].clear();
}

// submit_slot_writes

void GPUPartitioner::submit_slot_writes(int slot) {
    int active = 0;
    for (int p = 0; p < partition_count_; ++p)
        if (h_pinned_part_sizes_[slot][p] > 0) ++active;
    if (active == 0) return;

    slot_write_results_[slot].resize(active);
    slot_threads_[slot].clear();
    slot_threads_[slot].reserve(active);

    int task_idx = 0;
    for (int p = 0; p < partition_count_; ++p) {
        int sz = h_pinned_part_sizes_[slot][p];
        if (sz <= 0) continue;

        size_t bytes      = (size_t)sz * sizeof(int);
        off_t  file_off   = file_elem_offsets_[p] * (off_t)sizeof(int);
        off_t  buf_off    = (off_t)h_pinned_part_starts_[slot][p] * sizeof(int);
        file_elem_offsets_[p] += sz;

        int*           scatter_ptr = d_scatter_[slot];
        CUfileHandle_t handle      = partition_handles_[p];
        int            dev_id      = device_id_;
        WriteResult*   result      = &slot_write_results_[slot][task_idx++];
        result->requested = bytes;
        result->written   = 0;

        slot_threads_[slot].emplace_back(
            [dev_id, handle, scatter_ptr, bytes, file_off, buf_off, result]() {
                cudaSetDevice(dev_id);
                result->written = cuFileWrite(handle, scatter_ptr,
                                              bytes, file_off, buf_off);
            });
    }
}

// Main preprocessing entry point.

bool GPUPartitioner::preprocess_gpu_gds(bool isR, const int* data, size_t n) {

    {
        std::lock_guard<std::mutex> lk(g_stats_mutex);
        g_total_h2d_ms = g_total_compute_ms = g_total_scatter_ms =
        g_total_wait_ms = 0.0;
        g_total_batches = 0;
    }

    auto t_setup0 = std::chrono::steady_clock::now();
    checkCuda(cudaSetDevice(device_id_));
    checkCuda(cudaFree(0));
    double ctx_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_setup0).count();
    std::cout << "[smh] setup: cudaFree(0)/context=" << std::fixed << std::setprecision(1)
              << ctx_ms << "ms\n";

    for (int s = 0; s < NUM_SLOTS; ++s)
        wait_slot_writes(s);
    cleanup_file_resources();

    // Choose the batch size.
    size_t max_batch_elements = calculate_adaptive_batch_size(n);
    if (n > TARGET_BATCHES) {
        size_t chunk = (n + TARGET_BATCHES - 1) / TARGET_BATCHES;
        max_batch_elements = std::min(max_batch_elements, chunk);
    }
    const size_t MIN_BATCH = 1024ULL * 1024;
    if (max_batch_elements < MIN_BATCH && n > MIN_BATCH)
        max_batch_elements = std::min(MIN_BATCH, n);

    size_t expected_batches = (n + max_batch_elements - 1) / max_batch_elements;
    std::cout << "\n[smh] Processing " << n << " elements  "
              << "kernel=shared_mem_atomic (1 elem/thread)\n";
    std::cout << "[smh] max_batch=" << max_batch_elements
              << " (" << max_batch_elements * sizeof(int) / (1024.0*1024.0) << " MB)"
              << "  expected_batches=" << expected_batches << "\n";

    // Output directories and partition paths.
    std::string subdir = isR ? "file_R" : "file_S";
    for (const auto& base : base_paths_)
        std::filesystem::create_directories(base + "/" + subdir);

    int dir_cnt = (int)base_paths_.size();
    for (int i = 0; i < partition_count_; ++i)
        partition_files_[i] = base_paths_[i % dir_cnt] + "/" + subdir
                              + "/" + std::to_string(i) + ".bin";

    // Open, register, and preallocate partition files in parallel.
    size_t est_per_part = (size_t)((double)n / partition_count_ * 1.25) + 4096;
    size_t est_bytes    = align_up(est_per_part * sizeof(int), ALIGNMENT);

    {
        auto t_reg0 = std::chrono::steady_clock::now();
        std::vector<bool> reg_ok(partition_count_, true);
        std::vector<std::thread> reg_threads;
        reg_threads.reserve(partition_count_);
        for (int i = 0; i < partition_count_; ++i) {
            reg_threads.emplace_back([&, i]() {
                int fd = open(partition_files_[i].c_str(),
                              O_RDWR | O_CREAT | O_DIRECT, 0664);
                if (fd < 0) {
                    std::cerr << "[smh] open failed: " << partition_files_[i]
                              << ": " << strerror(errno) << "\n";
                    reg_ok[i] = false; return;
                }
                partition_fds_[i] = fd;
                if (ftruncate(fd, (off_t)est_bytes) != 0) {
                    std::cerr << "[smh] ftruncate failed for partition=" << i
                              << ": " << strerror(errno) << "\n";
                    reg_ok[i] = false; return;
                }
                CUfileDescr_t descr = {};
                descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
                descr.handle.fd = fd;
                if (cuFileHandleRegister(&partition_handles_[i], &descr).err != CU_FILE_SUCCESS) {
                    std::cerr << "[smh] cuFileHandleRegister failed for partition=" << i << "\n";
                    reg_ok[i] = false;
                }
            });
        }
        for (auto& t : reg_threads) t.join();
        double reg_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_reg0).count();
        std::cout << "[smh] Parallel file registration complete: " << std::fixed << std::setprecision(1)
                  << reg_ms << " ms  (preallocated per partition: "
                  << est_bytes / (1024.0*1024.0) << " MB)\n";
        for (int i = 0; i < partition_count_; ++i)
            if (!reg_ok[i]) return false;
    }

    // GPU compute buffers.
    if (pool_max_batch_size_ < max_batch_elements) {
        if (d_data_pool_)          { cudaFree(d_data_pool_);          d_data_pool_          = nullptr; }
        if (d_partitions_pool_)    { cudaFree(d_partitions_pool_);    d_partitions_pool_    = nullptr; }
        if (d_local_offsets_pool_) { cudaFree(d_local_offsets_pool_); d_local_offsets_pool_ = nullptr; }
        if (d_part_offsets_pool_)  { cudaFree(d_part_offsets_pool_);  d_part_offsets_pool_  = nullptr; }
        if (d_part_starts_pool_)   { cudaFree(d_part_starts_pool_);   d_part_starts_pool_   = nullptr; }

        pool_max_batch_size_ = max_batch_elements;
        auto t_malloc0 = std::chrono::steady_clock::now();
        checkCuda(cudaMalloc(&d_data_pool_,          max_batch_elements * sizeof(int)));
        checkCuda(cudaMalloc(&d_partitions_pool_,    max_batch_elements * sizeof(int)));
        checkCuda(cudaMalloc(&d_local_offsets_pool_, max_batch_elements * sizeof(int)));
        checkCuda(cudaMalloc(&d_part_offsets_pool_,  partition_count_   * sizeof(int)));
        checkCuda(cudaMalloc(&d_part_starts_pool_,   partition_count_   * sizeof(int)));
        double malloc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_malloc0).count();
        std::cout << "[smh] setup: cudaMalloc GPU compute bufs=" << std::fixed
                  << std::setprecision(1) << malloc_ms << "ms\n";
    }

    // Flat scatter buffers.
    size_t new_scatter_size = align_up(max_batch_elements * sizeof(int), ALIGNMENT);
    if (scatter_reg_size_ < new_scatter_size) {
        for (int s = 0; s < NUM_SLOTS; ++s) {
            if (d_scatter_[s]) {
                cuFileBufDeregister(d_scatter_[s]);
                cudaFree(d_scatter_[s]);
                d_scatter_[s] = nullptr;
            }
        }
        scatter_reg_size_ = new_scatter_size;

        std::cout << "[smh] Allocating and registering scatter buffers "
                  << scatter_reg_size_ / (1024.0*1024.0) << " MB x "
                  << NUM_SLOTS << " slots ...\n";

        auto t_buf0 = std::chrono::steady_clock::now();
        for (int s = 0; s < NUM_SLOTS; ++s) {
            checkCuda(cudaMalloc(&d_scatter_[s], scatter_reg_size_));
            CUfileError_t reg = cuFileBufRegister(d_scatter_[s], scatter_reg_size_, 0);
            if (reg.err != CU_FILE_SUCCESS) {
                fprintf(stderr, "[smh] cuFileBufRegister scatter[%d] failed err=%d\n",
                        s, (int)reg.err);
                return false;
            }
            double buf_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_buf0).count();
            std::cout << "[smh] setup: scatter[" << s << "] cudaMalloc+cuFileBufRegister="
                      << std::fixed << std::setprecision(1) << buf_ms << "ms (cumulative)\n";
        }
    }

    // Pinned host memory.
    for (int s = 0; s < NUM_SLOTS; ++s) {
        if (!h_pinned_part_sizes_[s])
            checkCuda(cudaMallocHost(&h_pinned_part_sizes_[s],
                                     partition_count_ * sizeof(int)));
        if (!h_pinned_part_starts_[s])
            checkCuda(cudaMallocHost(&h_pinned_part_starts_[s],
                                     partition_count_ * sizeof(int)));
    }

    // Compute stream.
    if (!compute_stream_)
        checkCuda(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));

    // GPU events used for timing.
    cudaEvent_t ev_h2d_s, ev_h2d_e, ev_comp_s, ev_comp_e, ev_scat_s, ev_scat_e;
    checkCuda(cudaEventCreate(&ev_h2d_s));  checkCuda(cudaEventCreate(&ev_h2d_e));
    checkCuda(cudaEventCreate(&ev_comp_s)); checkCuda(cudaEventCreate(&ev_comp_e));
    checkCuda(cudaEventCreate(&ev_scat_s)); checkCuda(cudaEventCreate(&ev_scat_e));

    // Main batch loop.

    auto wall_start = std::chrono::steady_clock::now();
    size_t processed = 0;
    int    batch_idx = 0;

    // Shared-memory size: one int per partition.
    size_t shared_mem_bytes = (size_t)partition_count_ * sizeof(int);

    while (processed < n) {
        int    slot       = batch_idx % NUM_SLOTS;
        size_t batch_size = std::min(max_batch_elements, n - processed);

        // Step 1: wait for the selected slot's previous writers.
        if (batch_idx >= NUM_SLOTS) {
            auto wt0 = std::chrono::steady_clock::now();
            wait_slot_writes(slot);
            double wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wt0).count();
            std::lock_guard<std::mutex> lk(g_stats_mutex);
            g_total_wait_ms += wait_ms;
        }

        // Step 2: host-to-device copy.
        checkCuda(cudaEventRecord(ev_h2d_s, compute_stream_));
        checkCuda(cudaMemcpyAsync(d_data_pool_,
                                  data + processed,
                                  batch_size * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream_));
        checkCuda(cudaEventRecord(ev_h2d_e, compute_stream_));

        // Step 3: shared-memory atomic-histogram partitioning.
        // Grid size is ceil(batch_size / 256); each thread processes one element.
        checkCuda(cudaMemsetAsync(d_part_offsets_pool_, 0,
                                  partition_count_ * sizeof(int), compute_stream_));
        checkCuda(cudaEventRecord(ev_comp_s, compute_stream_));

        int grid_compute = (int)((batch_size + 255) / 256);
        compute_partitions_and_offsets_kernel
            <<<grid_compute, 256, shared_mem_bytes, compute_stream_>>>(
                d_data_pool_, d_partitions_pool_, d_part_offsets_pool_,
                (uint64_t)batch_size, partition_count_,
                d_local_offsets_pool_);
        checkCuda(cudaGetLastError());
        checkCuda(cudaEventRecord(ev_comp_e, compute_stream_));

        // Step 4: exclusive prefix sum -> d_part_starts_pool_.
        exclusive_prefix_sum<<<1, 1, 0, compute_stream_>>>(
            d_part_offsets_pool_, d_part_starts_pool_, partition_count_);
        checkCuda(cudaGetLastError());

        // Step 5: convert local offsets to global flat offsets.
        int adj_grid = (int)((batch_size + 255) / 256);
        add_base_offsets<<<adj_grid, 256, 0, compute_stream_>>>(
            d_local_offsets_pool_, d_part_starts_pool_,
            d_partitions_pool_, (uint64_t)batch_size);
        checkCuda(cudaGetLastError());

        // Step 6: scatter elements into d_scatter_[slot].
        checkCuda(cudaEventRecord(ev_scat_s, compute_stream_));
        flat_scatter<<<adj_grid, 256, 0, compute_stream_>>>(
            d_data_pool_, d_local_offsets_pool_,
            d_scatter_[slot], (uint64_t)batch_size);
        checkCuda(cudaGetLastError());
        checkCuda(cudaEventRecord(ev_scat_e, compute_stream_));

        // Step 7: device-to-host copy of partition sizes and start offsets.
        checkCuda(cudaMemcpyAsync(h_pinned_part_sizes_[slot],
                                  d_part_offsets_pool_,
                                  partition_count_ * sizeof(int),
                                  cudaMemcpyDeviceToHost, compute_stream_));
        checkCuda(cudaMemcpyAsync(h_pinned_part_starts_[slot],
                                  d_part_starts_pool_,
                                  partition_count_ * sizeof(int),
                                  cudaMemcpyDeviceToHost, compute_stream_));

        // Step 8: the only CPU synchronization point.
        checkCuda(cudaStreamSynchronize(compute_stream_));

        float h2d_ms = 0, comp_ms = 0, scat_ms = 0;
        checkCuda(cudaEventElapsedTime(&h2d_ms,  ev_h2d_s,  ev_h2d_e));
        checkCuda(cudaEventElapsedTime(&comp_ms, ev_comp_s, ev_comp_e));
        checkCuda(cudaEventElapsedTime(&scat_ms, ev_scat_s, ev_scat_e));

        {
            std::lock_guard<std::mutex> lk(g_stats_mutex);
            g_total_h2d_ms     += h2d_ms;
            g_total_compute_ms += comp_ms;
            g_total_scatter_ms += scat_ms;
            ++g_total_batches;
        }

        // Step 9: submit background writers.
        submit_slot_writes(slot);

        processed += batch_size;
        std::cout << "[smh] batch " << batch_idx << " (slot " << slot << "): "
                  << batch_size << " elem  "
                  << "H2D=" << std::fixed << std::setprecision(1) << h2d_ms << "ms "
                  << "Comp=" << comp_ms << "ms "
                  << "Scat=" << scat_ms << "ms  | "
                  << std::setprecision(1) << 100.0 * processed / n << "%\n";

        ++batch_idx;
    }

    // Wait for the final batches' writers.
    {
        auto wt0 = std::chrono::steady_clock::now();
        for (int s = 0; s < NUM_SLOTS; ++s)
            wait_slot_writes(s);
        double wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wt0).count();
        std::lock_guard<std::mutex> lk(g_stats_mutex);
        g_total_wait_ms += wait_ms;
    }

    checkCuda(cudaEventDestroy(ev_h2d_s));  checkCuda(cudaEventDestroy(ev_h2d_e));
    checkCuda(cudaEventDestroy(ev_comp_s)); checkCuda(cudaEventDestroy(ev_comp_e));
    checkCuda(cudaEventDestroy(ev_scat_s)); checkCuda(cudaEventDestroy(ev_scat_e));

    auto wall_end = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(
        wall_end - wall_start).count();

    // Truncate each file to its actual written size.
    for (int i = 0; i < partition_count_; ++i) {
        off_t actual = (off_t)file_elem_offsets_[i] * sizeof(int);
        if (ftruncate(partition_fds_[i], actual) != 0)
            std::cerr << "Warning: ftruncate failed for partition=" << i << "\n";
    }

    int64_t total_elems = 0;
    for (int i = 0; i < partition_count_; ++i) total_elems += file_elem_offsets_[i];

    std::cout << "\nFinal partition sizes:\n";
    for (int i = 0; i < partition_count_; ++i)
        if (file_elem_offsets_[i] > 0)
            std::cout << "  Partition " << i << ": " << file_elem_offsets_[i] << " elements\n";
    std::cout << "  Total: " << total_elems << " elements\n"
              << "  Batches: " << g_total_batches << "\n";

    if (g_total_batches > 0) {
        double gpu_total = g_total_h2d_ms + g_total_compute_ms + g_total_scatter_ms;
        double h2d_bw_gbs = (n * sizeof(int)) / (g_total_h2d_ms / 1000.0) / 1e9;
        std::cout << "\n=== v11-diff Ping-Pong timing summary (shared-memory atomic kernel) ===\n"
                  << "  H2D (cumulative):      " << std::fixed << std::setprecision(1)
                                             << g_total_h2d_ms     << " ms"
                  << "  [H2D BW=" << std::setprecision(2) << h2d_bw_gbs << " GB/s"
                  << "  (expected pinned bandwidth >=12 GB/s; pageable memory about 9 GB/s)]\n"
                  << "  Compute (cumulative):  " << std::fixed << std::setprecision(1)
                                             << g_total_compute_ms << " ms\n"
                  << "  Scatter (cumulative):  " << g_total_scatter_ms << " ms\n"
                  << "  GPU compute total:     " << gpu_total           << " ms\n"
                  << "  Write wait:            " << g_total_wait_ms
                  << " ms  (near zero: I/O is faster than GPU; large: I/O is the bottleneck)\n"
                  << "\n  Wall time: " << wall_ms << " ms\n"
                  << "  Throughput: "
                  << std::setprecision(2)
                  << (n / (wall_ms / 1000.0)) / 1e6 << " M items/s\n";

        double ideal_parallel = gpu_total / g_total_batches;
        double wait_ratio     = g_total_wait_ms / wall_ms;
        std::cout << "\n  [Bottleneck classification]\n"
                  << "  Average GPU time per batch: " << ideal_parallel << " ms\n"
                  << "  Wait share of wall time:    " << std::setprecision(1) << wait_ratio * 100 << "%\n"
                  << "  -> " << (wait_ratio > 0.3
                      ? "I/O remains the bottleneck; consider more slots or disks"
                      : "I/O and GPU work overlap well") << "\n";
    }

    return true;
}
