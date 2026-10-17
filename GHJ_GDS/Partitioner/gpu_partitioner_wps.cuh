/*
gpu_partitioner_wps.cuh - partitioner interface (warp-shuffle prefix sum)

Included by GHJ_GDS_WPS. Public interface:
  GPUPartitioner(partition_count, device_id, base_paths, max_batch_size)
  bool preprocess_gpu_gds(bool isR, const int* data, size_t n)
  set_items_per_thread(int) adjusts the number of input items per thread.
The input interface is (isR, data, n) and accepts pinned host memory.
*/
#ifndef GPU_PARTITIONER_WPS_H
#define GPU_PARTITIONER_WPS_H

// Ping-pong cuFile writes with parallel file registration and pinned host input.

#include <vector>
#include <thread>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include <cufile.h>

class GPUPartitioner {
public:
    GPUPartitioner(int partition_count, int device_id = 0,
                   const std::vector<std::string>& base_paths = {},
                   size_t max_batch_size = 1000000000ULL);
    ~GPUPartitioner();

    void set_items_per_thread(int val);
    bool preprocess_gpu_gds(bool isR, const int* data, size_t n);

private:
    // Basic configuration.
    int    partition_count_;
    int    device_id_;
    int    items_per_thread_;
    size_t max_batch_size_;
    std::vector<std::string> base_paths_;

    // File resources.
    std::vector<std::string>    partition_files_;
    std::vector<int>            partition_fds_;
    std::vector<CUfileHandle_t> partition_handles_;
    std::vector<int64_t>        file_elem_offsets_;

    // GPU compute buffers. A single compute stream avoids internal contention.
    int*   d_data_pool_          = nullptr;
    int*   d_partitions_pool_    = nullptr;
    int*   d_local_offsets_pool_ = nullptr;
    int*   d_part_offsets_pool_  = nullptr;
    int*   d_part_starts_pool_   = nullptr;
    size_t pool_max_batch_size_  = 0;

    // Ping-pong slots.
    static constexpr int NUM_SLOTS = 2;

    int*   d_scatter_[NUM_SLOTS]      = {nullptr, nullptr};
    size_t scatter_reg_size_          = 0;

    int*   h_pinned_part_sizes_[NUM_SLOTS]  = {nullptr, nullptr};
    int*   h_pinned_part_starts_[NUM_SLOTS] = {nullptr, nullptr};

    cudaStream_t compute_stream_ = nullptr;

    // Background write-thread management.
    std::vector<std::thread>   slot_threads_[NUM_SLOTS];
    struct WriteResult { ssize_t written; size_t requested; };
    std::vector<WriteResult>   slot_write_results_[NUM_SLOTS];

    // Helper functions.
    void   cleanup_persistent_resources();
    void   cleanup_file_resources();
    size_t calculate_adaptive_batch_size(size_t remaining_elements);
    void   wait_slot_writes(int slot);
    void   submit_slot_writes(int slot);

    // Deprecated interface stubs.
    struct PartitionBuffers { std::vector<int*> d_buffers; std::vector<int> sizes; };
    void preallocate_partition_buffers(size_t) {}
    void ensure_partition_buffer(int, size_t) {}
    void ensure_device_pointer_array() {}
    void compute_optimized_partitions_gpu(int*, int*, int*, int*, uint64_t) {}
    void validate_partition_offsets(int*, uint64_t) {}
    PartitionBuffers organize_optimized_partitions_gpu(int*, int*, int*, int*, uint64_t)
    { return {}; }
    bool process_batch(bool, const std::vector<int>&, size_t, size_t, std::vector<int>&)
    { return false; }
    void write_batch_partitions_gds(const PartitionBuffers&, std::vector<int>&) {}
    void write_single_partition_batch_gds(int, int*, int64_t, int) {}
    void cleanup_batch_resources() {}
    void cleanup_resources() {}
};

#endif // GPU_PARTITIONER_WPS_H
