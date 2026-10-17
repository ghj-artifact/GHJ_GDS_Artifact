/*
gpu_hash_join.cu - join implementation with a counting hash table,
per-directory read lanes, and a configurable read structure.

Build: compile as a translation unit of the main program.
Entry point: GHJ_GDS_JOIN(base_paths, partition_count, device_id,
num_workers, dir_max_streams).

Implementation:
  - Hash table: open-addressing key and count tables with O(R+S) work,
    independent of Zipf skew. Each directory has a bounded number of in-flight reads.
  - Reads: cuFileRead copies data into per-worker buffers; build and probe run on the GPU.
  - Read structure is selected through environment variables:
      GHJ_CHUNK_MB      0 or unset = read whole partitions; >0 = chunk size in MiB
      GHJ_CHUNK_READERS number of chunks read concurrently per round (default 1)
      GHJ_PREFETCH      1 = cross-partition prefetch (default); 0 = disabled
      GHJ_RS_PARALLEL   1 = read R and S concurrently for a partition (default 0)
    CHUNK=0/PREFETCH=1 matches the earlier whole-partition prefetch path;
    CHUNK=0/PREFETCH=0 matches the non-prefetch path; CHUNK>0 uses the chunked path.

Chunked reads and cross-partition prefetching are mutually exclusive because
chunk buffers are reused each round. Prefetch is disabled when both are set.

Test requirements:
  - Select the intended GPU and set CUDA_MPS_PIPE_DIRECTORY to an isolated
    directory when a shared MPS daemon may be present.
  - Use 4 MiB cuFile I/O chunks:
    export CUFILE_ENV_PATH_JSON=<artifact-root>/GHJ_GDS/cufile_gds4m.json
  - Every probe must start after all build work completes.
  - Identical inputs must produce the same total_result.
*/

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "gpu_hash_join.cuh"
#include <iostream>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <deque>
#include <map>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cufile.h>
#include "../types_GDS.h"

// Global timing aggregation for the join phase.
// Each partition accumulates readR, readS, build, and probe work. The two
// non-overlapping stage estimates are:
//   stageA = sum(max(readR_i, build_i))  // R read overlaps build
//   stageB = sum(max(readS_i, probe_i))  // S read overlaps probe
// stageA + stageB approximately equals join_wall. Do not stack read and compute
// work directly because that double-counts overlapped intervals.
static std::mutex g_js_mu;
static double g_js_readR = 0, g_js_readS = 0, g_js_build = 0, g_js_probe = 0;
static double g_js_stageA = 0, g_js_stageB = 0;
static double g_js_setup = 0, g_js_tail = 0;   // Measured setup and tail time in milliseconds.
static int    g_js_parts = 0;

#define checkCuda(err) __checkCuda(err, __FILE__, __LINE__)
inline void __checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s at %s:%d\n",
                cudaGetErrorString(result), file, line);
        assert(result == cudaSuccess);
    }
}

typedef unsigned long long ull;

#define CHECK_CUFILE(call)                                               \
do {                                                                     \
    CUfileError_t err = call;                                            \
    if (err.err != CU_FILE_SUCCESS) {                                    \
        fprintf(stderr, "cuFile error at %s:%d - Error Code: %d\n",     \
                __FILE__, __LINE__, err.err);                            \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

static int NUM_WORKERS  = 4;   // runtime parameter (clamped automatically)
static const size_t PAGE_SIZE = 4096;

// Every valid key lies in [0, data_size-1], so -1 is safe as the empty marker
static constexpr int32_t EMPTY_KEY = -1;

static std::vector<std::string> g_partition_R_paths;
static std::vector<std::string> g_partition_S_paths;
static std::vector<size_t> g_R_bytes;
static std::vector<size_t> g_S_bytes;

static std::mutex            queue_mutex;
static std::condition_variable cv;
// Per-directory read lanes.
struct DirLane {
    std::deque<int> pending;
    int active = 0;
    int limit  = 1;
};
static std::vector<int>     g_partition_lane;
static std::vector<DirLane> g_lanes;
static std::atomic<ull>      total_result(0);
static std::atomic<int>      active_workers(0);
static std::atomic<bool>     all_done(false);
static std::atomic<int>      completed_partitions(0);
static int total_partitions = 0;

// Read-structure options.
static size_t g_chunk_bytes  = 0;      // 0 = read whole partitions.
static size_t g_chunk_elems  = 0;
static int    g_n_readers    = 1;      // Concurrent chunks per read round.
static int    g_prefetch     = 1;      // 1 = cross-partition prefetch.
static int    g_rs_parallel  = 0;      // 1 = concurrently read R and S.
static int    g_chunked      = 0;      // Derived: whether chunked reads are enabled.
static int    g_prefetch_eff = 1;      // Derived: effective prefetch setting.

static void init_read_config() {
    if (const char* e = getenv("GHJ_CHUNK_MB")) {
        long long mb = atoll(e);
        if (mb > 0) g_chunk_bytes = (size_t)mb * 1024 * 1024;
    }
    if (const char* r = getenv("GHJ_CHUNK_READERS")) {
        int v = atoi(r);
        g_n_readers = (v > 0) ? v : 1;
    }
    if (const char* p = getenv("GHJ_PREFETCH")) {
        g_prefetch = (atoi(p) != 0) ? 1 : 0;
    }
    if (const char* s = getenv("GHJ_RS_PARALLEL")) {
        g_rs_parallel = (atoi(s) != 0) ? 1 : 0;
    }
    g_chunked = (g_chunk_bytes > 0) ? 1 : 0;
    g_prefetch_eff = g_prefetch;
    if (g_chunked && g_prefetch) {
        printf("[join] Chunked reads and cross-partition prefetch are mutually exclusive; setting GHJ_PREFETCH=0\n");
        g_prefetch_eff = 0;
    }
    if (g_chunked) {
        size_t elems = g_chunk_bytes / sizeof(intkey_t);
        elems &= ~(size_t)1023;            // Align reads to 1024 elements (4096 bytes).
        if (elems < 1024) elems = 1024;
        g_chunk_elems = elems;
        printf("[join] Read structure: chunked, GHJ_CHUNK_MB=%zu (%zu elements / %zu bytes per chunk, %d readers per round)"
               " | prefetch=forced off | parallel R/S reads=%s\n",
               g_chunk_bytes / (1024 * 1024), g_chunk_elems, g_chunk_elems * sizeof(intkey_t),
               g_n_readers, g_rs_parallel ? "on" : "off");
    } else {
        g_chunk_elems = (size_t)1 << 62;   // A single read can cover the whole partition.
        printf("[join] Read structure: whole partition | cross-partition prefetch=%s | parallel R/S reads=%s\n",
               g_prefetch_eff ? "on" : "off", g_rs_parallel ? "on" : "off");
    }
}

// Murmur3 hash.
__device__ __host__ inline uint32_t murmur3_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

// Build kernel: open-addressing counting hash table
__global__ void build_count_table(int32_t* __restrict__ key_table,
                                       int32_t* __restrict__ cnt_table,
                                       size_t table_size,
                                       const intkey_t* __restrict__ R,
                                       size_t R_n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R_n) return;

    intkey_t key  = __ldg(&R[i]);
    uint32_t h    = murmur3_hash((uint32_t)key);
    int      slot = (int)(h % (uint32_t)table_size);

    while (true) {
        int32_t old = atomicCAS(&key_table[slot], EMPTY_KEY, key);
        if (old == EMPTY_KEY || old == key) {
            // claimed the slot, or found an existing slot with the same key
            atomicAdd(&cnt_table[slot], 1);
            return;
        }
        // the slot holds another key, probe the next one
        slot = (slot + 1 < table_size) ? slot + 1 : 0;
    }
}

// Probe kernel: look up S elements in the counting table and add the matching R counts
__global__ void probe_count_table(const int32_t* __restrict__ key_table,
                                       const int32_t* __restrict__ cnt_table,
                                       size_t table_size,
                                       const intkey_t* __restrict__ S,
                                       size_t S_n,
                                       ull* __restrict__ matches)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    ull local = 0;

    if (i < S_n) {
        intkey_t key  = __ldg(&S[i]);
        uint32_t h    = murmur3_hash((uint32_t)key);
        int      slot = (int)(h % (uint32_t)table_size);

        while (true) {
            int32_t k = key_table[slot];
            if (k == EMPTY_KEY) break;   // the key is not in R
            if (k == key) {
                local = (ull)(uint32_t)cnt_table[slot];
                break;
            }
            slot = (slot + 1 < table_size) ? slot + 1 : 0;
        }
    }

    // Warp-level reduction to cut atomicAdd contention
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(mask, local, offset);
    if ((threadIdx.x & 31) == 0 && local > 0)
        atomicAdd(matches, local);
}

// Worker: owns the counting hash table buffers
struct Worker {
    int id;
    int device_id;

    cudaStream_t streamCompute;

    intkey_t* R_numbers;
    intkey_t* S_numbers;
    ull*      g_matches;

    int32_t* d_key_table;   // open-addressing key table (initialised to EMPTY_KEY = -1)
    int32_t* d_cnt_table;   // count table (initialised to 0)
    size_t   table_size;    // = max_R_elements * HASH_SCALE_FACTOR

    size_t max_R_bytes_aligned;   // Largest R partition size in bytes; determines table size.
    size_t max_S_bytes_aligned;
    size_t slot_R_bytes;          // R buffer per slot: one partition or two rounds of chunks.
    size_t slot_S_bytes;          // S buffer per slot; S uses whole-partition buffers in parallel mode.
    size_t buf_R_bytes;           // Allocated size: slot_R_bytes times the number of slots.
    size_t buf_S_bytes;
    intkey_t* R_slot[2] = {nullptr, nullptr};   // Two slots; slot 1 is unused without prefetch.
    intkey_t* S_slot[2] = {nullptr, nullptr};

    Worker(int wid, int dev, size_t max_R_bytes, size_t max_S_bytes)
        : id(wid), device_id(dev), streamCompute(nullptr),
          R_numbers(nullptr), S_numbers(nullptr), g_matches(nullptr),
          d_key_table(nullptr), d_cnt_table(nullptr), table_size(0) {

        cudaSetDevice(device_id);
        checkCuda(cudaStreamCreate(&streamCompute));

        max_R_bytes_aligned = (max_R_bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        max_S_bytes_aligned = (max_S_bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

        // Buffer layout depends on the read options.
        // Whole-partition mode: one partition per slot.
        // Chunked mode: two rounds of NR chunks per slot. In parallel R/S mode,
        // S uses a whole-partition buffer that remains available for probe.
        const size_t ring = (size_t)(2 * g_n_readers) * (g_chunk_bytes ? g_chunk_bytes : PAGE_SIZE);
        slot_R_bytes = g_chunked ? std::min(ring, max_R_bytes_aligned) : max_R_bytes_aligned;
        slot_S_bytes = g_chunked ? (g_rs_parallel ? max_S_bytes_aligned
                                                  : std::min(ring, max_S_bytes_aligned))
                                 : max_S_bytes_aligned;
        slot_R_bytes = (slot_R_bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        slot_S_bytes = (slot_S_bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        if (slot_R_bytes == 0) slot_R_bytes = PAGE_SIZE;
        if (slot_S_bytes == 0) slot_S_bytes = PAGE_SIZE;
        const int n_slots = (g_prefetch_eff || !g_chunked) ? 2 : 1;   // One slot for chunked mode without prefetch.
        buf_R_bytes = slot_R_bytes * n_slots;
        buf_S_bytes = slot_S_bytes * n_slots;

        checkCuda(cudaMalloc(&R_numbers, buf_R_bytes));
        checkCuda(cudaMalloc(&S_numbers, buf_S_bytes));
        checkCuda(cudaMalloc(&g_matches, sizeof(ull)));
        R_slot[0] = R_numbers;                        R_slot[1] = R_numbers + slot_R_bytes / sizeof(intkey_t);
        S_slot[0] = S_numbers;                        S_slot[1] = S_numbers + slot_S_bytes / sizeof(intkey_t);

        size_t max_R_elements = max_R_bytes / sizeof(intkey_t);
        table_size = max_R_elements * HASH_SCALE_FACTOR;
        checkCuda(cudaMalloc(&d_key_table, (size_t)table_size * sizeof(int32_t)));
        checkCuda(cudaMalloc(&d_cnt_table, (size_t)table_size * sizeof(int32_t)));

        // cuFileBufRegister: cuFileRead needs a registered GPU buffer
        CUfileError_t status;
        status = cuFileBufRegister(R_numbers, buf_R_bytes, 0);
        if (status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "Worker %d: cuFileBufRegister R failed: %d\n", id, status.err);
            exit(EXIT_FAILURE);
        }
        status = cuFileBufRegister(S_numbers, buf_S_bytes, 0);
        if (status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "Worker %d: cuFileBufRegister S failed: %d\n", id, status.err);
            exit(EXIT_FAILURE);
        }
    }

    ~Worker() {
        if (R_numbers)   { cuFileBufDeregister(R_numbers); cudaFree(R_numbers); }
        if (S_numbers)   { cuFileBufDeregister(S_numbers); cudaFree(S_numbers); }
        if (g_matches)   cudaFree(g_matches);
        if (d_key_table) cudaFree(d_key_table);
        if (d_cnt_table) cudaFree(d_cnt_table);
        if (streamCompute) cudaStreamDestroy(streamCompute);
    }

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
};

// Read one complete partition side with one cuFileRead. With prefetching, the
// two slots alternate between consecutive partitions.
static void read_side(intkey_t* buf, const std::string& path, size_t bytes,
                      int worker_id, const char* side, double* out_ms) {
    CUfileDescr_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) { fprintf(stderr, "Worker %d: open %s failed: %s\n", worker_id, side, strerror(errno)); return; }
    CUfileHandle_t fh = nullptr;
    desc.handle.fd = fd;
    if (cuFileHandleRegister(&fh, &desc).err != CU_FILE_SUCCESS) { close(fd); return; }
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    ssize_t rd = cuFileRead(fh, buf, bytes, 0, 0);
    if (out_ms) *out_ms += duration<double, std::milli>(steady_clock::now() - t0).count();
    if (rd < 0 || (size_t)rd != bytes)
        fprintf(stderr, "Worker %d: cuFileRead %s failed/incomplete: %zd/%zu\n", worker_id, side, rd, bytes);
    cuFileHandleDeregister(fh);
    close(fd);
}

static bool try_pick_partition_locked(int& pid) {
    int best = -1;
    for (size_t i = 0; i < g_lanes.size(); i++) {
        const DirLane& l = g_lanes[i];
        if (l.pending.empty() || l.active >= l.limit) continue;
        if (best < 0) { best = (int)i; continue; }
        const DirLane& b = g_lanes[best];
        if (l.limit < b.limit || (l.limit == b.limit && l.pending.size() > b.pending.size())) best = (int)i;
    }
    if (best < 0) return false;
    DirLane& l = g_lanes[best];
    pid = l.pending.front(); l.pending.pop_front(); l.active++;
    return true;
}

static void worker_function(Worker* worker) {
    cudaSetDevice(worker->device_id);
    active_workers++;
    using namespace std::chrono;
    const int threadsPerBlock = 256;
    cudaEvent_t e_b0, e_b1, e_p0, e_p1;
    cudaEvent_t ev_round[2];                  // Protect chunk-buffer reuse between rounds.
    checkCuda(cudaEventCreate(&e_b0)); checkCuda(cudaEventCreate(&e_b1));
    checkCuda(cudaEventCreate(&e_p0)); checkCuda(cudaEventCreate(&e_p1));
    checkCuda(cudaEventCreate(&ev_round[0])); checkCuda(cudaEventCreate(&ev_round[1]));

    auto pick = [&](int& pid) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [&]() {
            if (all_done.load()) return true;
            return try_pick_partition_locked(pid);
        });
        lock.unlock();
        cv.notify_all();
        return pid >= 0;
    };
    auto release = [&](int pid) {
        if (pid < 0) return;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            g_lanes[g_partition_lane[pid]].active--;
        }
        cv.notify_all();
        ++completed_partitions;
    };
    // Prefetch selection must be non-blocking. Return false immediately when the
    // queue is empty or the worker's lane is full; otherwise the prefetch path and
    // the completion condition can wait on each other and deadlock.
    auto try_pick_now = [&](int& pid) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        bool ok = try_pick_partition_locked(pid);
        cv.notify_all();
        return ok;
    };

    // Read-structure parameters.
    const int    NR          = g_n_readers;
    const int    NRB         = 2;
    const bool   chunked     = (g_chunked != 0);
    const bool   rs_par      = (g_rs_parallel != 0);
    const bool   prefetch    = (g_prefetch_eff != 0);
    const size_t chunk_elems = g_chunk_elems;

    // Whole-partition parallel R/S reads use one handle and thread per side.
    auto read_whole_pair = [&](int pid, int sl, double* r_out, double* s_out) {
        double rl = 0.0, sl_ms = 0.0;
        std::thread tR([&]() { cudaSetDevice(worker->device_id);
            read_side(worker->R_slot[sl], g_partition_R_paths[pid], g_R_bytes[pid],
                      worker->id, "R", &rl); });
        std::thread tS([&]() { cudaSetDevice(worker->device_id);
            read_side(worker->S_slot[sl], g_partition_S_paths[pid], g_S_bytes[pid],
                      worker->id, "S", &sl_ms); });
        tR.join(); tS.join();
        *r_out += rl; *s_out += sl_ms;
    };

    int a = -1;
    if (!pick(a)) { active_workers--; return; }

    // Prologue for whole-partition mode with prefetching: load the first
    // partition's R and S data before entering the main loop. Without prefetch,
    // each partition is read in loop order: R, build, S, probe.
    double r_tot = 0.0, s_tot = 0.0;
    bool R_ready = false, S_ready = false;     // Whether partition a's R/S data is already buffered.
    if (!chunked && prefetch) {
        if (rs_par) {
            read_whole_pair(a, 0, &r_tot, &s_tot);
            R_ready = S_ready = true;
        } else {
            read_side(worker->R_slot[0], g_partition_R_paths[a], g_R_bytes[a], worker->id, "R", &r_tot);
            read_side(worker->S_slot[0], g_partition_S_paths[a], g_S_bytes[a], worker->id, "S", &s_tot);
            R_ready = S_ready = true;
        }
    }
    int slot = 0;

    while (a >= 0) {
        int b = -1;
        if (prefetch) try_pick_now(b);             // Non-blocking prefetch target.
        auto t_start = steady_clock::now();
        const size_t R_el = g_R_bytes[a] / sizeof(intkey_t);
        const size_t S_el = g_S_bytes[a] / sizeof(intkey_t);
        const size_t ht   = R_el * HASH_SCALE_FACTOR;
        const int next_slot = prefetch ? (slot ^ 1) : slot;
        double r_ms = 0.0, s_ms = 0.0;
        // Measure setup, stage A, stage B, and tail time for the breakdown report.
        double wallA_ms = 0.0, wallB_ms = 0.0, setup_ms = 0.0, tail_ms = 0.0;

        if (!chunked) {
            // Whole-partition path: build(a), prefetch R(b), probe(a), prefetch S(b).
            // The R side, and optionally S, may already be loaded by the prologue
            // or the previous prefetch.
            if (!R_ready) {
                if (rs_par) { read_whole_pair(a, slot, &r_ms, &s_ms); S_ready = true; }
                else read_side(worker->R_slot[slot], g_partition_R_paths[a], g_R_bytes[a],
                               worker->id, "R", &r_ms);
                R_ready = true;
            }
            // Initialize the counting tables before build on the same stream.
            checkCuda(cudaMemsetAsync(worker->d_key_table, 0xFF, ht * sizeof(int32_t), worker->streamCompute));
            checkCuda(cudaMemsetAsync(worker->d_cnt_table, 0x00, ht * sizeof(int32_t), worker->streamCompute));
            checkCuda(cudaEventRecord(e_b0, worker->streamCompute));
            build_count_table<<<(R_el + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0,
                                    worker->streamCompute>>>(
                worker->d_key_table, worker->d_cnt_table, ht, worker->R_slot[slot], R_el);
            checkCuda(cudaGetLastError());
            checkCuda(cudaEventRecord(e_b1, worker->streamCompute));

            // Prefetch b's R side, and optionally S, while build(a) runs.
            if (b >= 0) {
                if (rs_par) read_whole_pair(b, next_slot, &r_ms, &s_ms);
                else read_side(worker->R_slot[next_slot], g_partition_R_paths[b], g_R_bytes[b],
                               worker->id, "R", &r_ms);
            }
            // Without prefetch, read this partition's S side after build(a).
            else if (!S_ready) {
                read_side(worker->S_slot[slot], g_partition_S_paths[a], g_S_bytes[a],
                          worker->id, "S", &s_ms);
            }

            // Probe(a) is asynchronous; stream order places it after every build.
            checkCuda(cudaMemsetAsync(worker->g_matches, 0, sizeof(ull), worker->streamCompute));
            checkCuda(cudaEventRecord(e_p0, worker->streamCompute));
            probe_count_table<<<(S_el + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0,
                                    worker->streamCompute>>>(
                worker->d_key_table, worker->d_cnt_table, ht, worker->S_slot[slot], S_el, worker->g_matches);
            checkCuda(cudaGetLastError());
            checkCuda(cudaEventRecord(e_p1, worker->streamCompute));

            // Prefetch b's S side while probe(a) runs. Parallel mode already loaded it.
            if (b >= 0 && !rs_par)
                read_side(worker->S_slot[next_slot], g_partition_S_paths[b], g_S_bytes[b],
                          worker->id, "S", &s_ms);
        } else {
            // Chunked path: stage A overlaps R reads with build; stage B overlaps
            // S reads with probe.
            checkCuda(cudaMemsetAsync(worker->d_key_table, 0xFF, ht * sizeof(int32_t), worker->streamCompute));
            checkCuda(cudaMemsetAsync(worker->d_cnt_table, 0x00, ht * sizeof(int32_t), worker->streamCompute));
            checkCuda(cudaEventRecord(e_b0, worker->streamCompute));
            size_t n_R_chunks = (R_el + chunk_elems - 1) / chunk_elems; if (n_R_chunks == 0) n_R_chunks = 1;
            size_t n_S_chunks = (S_el + chunk_elems - 1) / chunk_elems; if (n_S_chunks == 0) n_S_chunks = 1;
            const size_t n_R_rounds = (n_R_chunks + (size_t)NR - 1) / (size_t)NR;
            const size_t n_S_rounds = (n_S_chunks + (size_t)NR - 1) / (size_t)NR;
            size_t S_prefetched = 0;

            // Chunked mode keeps handles open across rounds.
            CUfileDescr_t desc; memset(&desc, 0, sizeof(desc));
            desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            CUfileHandle_t R_h = nullptr, S_h = nullptr;
            int R_fd = open(g_partition_R_paths[a].c_str(), O_RDONLY | O_DIRECT);
            int S_fd = open(g_partition_S_paths[a].c_str(), O_RDONLY | O_DIRECT);
            if (R_fd < 0 || S_fd < 0 ||
                (desc.handle.fd = R_fd, cuFileHandleRegister(&R_h, &desc).err) != CU_FILE_SUCCESS ||
                (desc.handle.fd = S_fd, cuFileHandleRegister(&S_h, &desc).err) != CU_FILE_SUCCESS) {
                fprintf(stderr, "Worker %d: failed to open R/S files for partition %d: %s\n", worker->id, a, strerror(errno));
                if (R_h) cuFileHandleDeregister(R_h);
                if (S_h) cuFileHandleDeregister(S_h);
                if (R_fd >= 0) close(R_fd);
                if (S_fd >= 0) close(S_fd);
                release(a);
                slot = 0; a = -1;
                if (!prefetch && !pick(a)) break;
                continue;
            }

            intkey_t* Rbuf = worker->R_slot[slot];
            intkey_t* Sbuf = worker->S_slot[slot];
            auto R_off = [&](size_t r) -> size_t { return (r % (size_t)NRB) * (size_t)NR * chunk_elems; };
            auto S_off = [&](size_t r) -> size_t {
                return rs_par ? (r * (size_t)NR * chunk_elems)
                              : ((r % (size_t)NRB) * (size_t)NR * chunk_elems);
            };
            auto read_round = [&](CUfileHandle_t fh, intkey_t* buf0, size_t round_idx, size_t n_chunks,
                                  size_t total_elems, const char* side, double* out_ms) {
                const size_t first = round_idx * (size_t)NR;
                const int    m     = (int)std::min<size_t>((size_t)NR, n_chunks - first);
                std::vector<std::thread> th; std::vector<char> ok(m, 1);
                th.reserve(m);
                auto t0 = steady_clock::now();
                for (int i = 0; i < m; i++) {
                    th.emplace_back([&, i]() {
                        cudaSetDevice(worker->device_id);
                        const size_t off = (first + (size_t)i) * chunk_elems;
                        const size_t n   = std::min(chunk_elems, total_elems - off);
                        ssize_t rd = cuFileRead(fh, buf0 + (size_t)i * chunk_elems, n * sizeof(intkey_t),
                                                (off_t)(off * sizeof(intkey_t)), 0);
                        if (rd < 0 || (size_t)rd != n * sizeof(intkey_t)) ok[i] = 0;
                    });
                }
                for (auto& t : th) t.join();
                *out_ms += duration<double, std::milli>(steady_clock::now() - t0).count();
                for (int i = 0; i < m; i++)
                    if (!ok[i]) fprintf(stderr, "Worker %d: cuFileRead %s chunk %zu failed/incomplete\n",
                                        worker->id, side, first + (size_t)i);
            };
            // Run one R round and one S round together through two handles.
            auto read_round_pair = [&](size_t rR, size_t rS, double* out_ms) {
                const size_t fR = rR * (size_t)NR, fS = rS * (size_t)NR;
                const int    mR = (int)std::min<size_t>((size_t)NR, n_R_chunks - fR);
                const int    mS = (int)std::min<size_t>((size_t)NR, n_S_chunks - fS);
                intkey_t*    bR = Rbuf + R_off(rR);
                intkey_t*    bS = Sbuf + S_off(rS);
                std::vector<std::thread> th; std::vector<char> okR(mR, 1), okS(mS, 1);
                th.reserve((size_t)(mR + mS));
                auto t0 = steady_clock::now();
                for (int i = 0; i < mR; i++)
                    th.emplace_back([&, i]() {
                        cudaSetDevice(worker->device_id);
                        const size_t off = (fR + (size_t)i) * chunk_elems;
                        const size_t n   = std::min(chunk_elems, R_el - off);
                        ssize_t rd = cuFileRead(R_h, bR + (size_t)i * chunk_elems, n * sizeof(intkey_t),
                                                (off_t)(off * sizeof(intkey_t)), 0);
                        if (rd < 0 || (size_t)rd != n * sizeof(intkey_t)) okR[i] = 0;
                    });
                for (int i = 0; i < mS; i++)
                    th.emplace_back([&, i]() {
                        cudaSetDevice(worker->device_id);
                        const size_t off = (fS + (size_t)i) * chunk_elems;
                        const size_t n   = std::min(chunk_elems, S_el - off);
                        ssize_t rd = cuFileRead(S_h, bS + (size_t)i * chunk_elems, n * sizeof(intkey_t),
                                                (off_t)(off * sizeof(intkey_t)), 0);
                        if (rd < 0 || (size_t)rd != n * sizeof(intkey_t)) okS[i] = 0;
                    });
                for (auto& t : th) t.join();
                *out_ms += duration<double, std::milli>(steady_clock::now() - t0).count();
                for (int i = 0; i < mR; i++) if (!okR[i])
                    fprintf(stderr, "Worker %d: cuFileRead R chunk %zu failed/incomplete (pair)\n",
                            worker->id, fR + (size_t)i);
                for (int i = 0; i < mS; i++) if (!okS[i])
                    fprintf(stderr, "Worker %d: cuFileRead S chunk %zu failed/incomplete (pair)\n",
                            worker->id, fS + (size_t)i);
            };

            // Stage A: build from R chunks.
            auto tA0 = steady_clock::now();
            setup_ms = duration<double, std::milli>(tA0 - t_start).count();
            if (rs_par && n_S_rounds > 0) { read_round_pair(0, 0, &r_ms); S_prefetched = 1; }
            else                            read_round(R_h, Rbuf + R_off(0), 0, n_R_chunks, R_el, "R", &r_ms);
            for (size_t r = 0; r < n_R_rounds; r++) {
                const size_t first = r * (size_t)NR;
                const int    m     = (int)std::min<size_t>((size_t)NR, n_R_chunks - first);
                intkey_t*    buf0  = Rbuf + R_off(r);
                for (int i = 0; i < m; i++) {
                    const size_t n = std::min(chunk_elems, R_el - (first + (size_t)i) * chunk_elems);
                    build_count_table<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0,
                                            worker->streamCompute>>>(
                        worker->d_key_table, worker->d_cnt_table, ht, buf0 + (size_t)i * chunk_elems, n);
                    checkCuda(cudaGetLastError());
                }
                checkCuda(cudaEventRecord(ev_round[r % NRB], worker->streamCompute));
                if (r + 1 < n_R_rounds) {
                    if (r >= 1) checkCuda(cudaEventSynchronize(ev_round[(r + 1) % NRB]));
                    if (rs_par && r + 1 < n_S_rounds) { read_round_pair(r + 1, r + 1, &r_ms); S_prefetched = r + 2; }
                    else read_round(R_h, Rbuf + R_off(r + 1), r + 1, n_R_chunks, R_el, "R", &r_ms);
                }
            }
            checkCuda(cudaEventRecord(e_b1, worker->streamCompute));   // Completion point for all build work.
            auto tA1 = steady_clock::now();
            wallA_ms = duration<double, std::milli>(tA1 - tA0).count();

            // Stage B: probe S chunks.
            checkCuda(cudaMemsetAsync(worker->g_matches, 0, sizeof(ull), worker->streamCompute));
            if (S_prefetched < n_S_rounds)
                read_round(S_h, Sbuf + S_off(S_prefetched), S_prefetched, n_S_chunks, S_el, "S", &s_ms);
            checkCuda(cudaEventRecord(e_p0, worker->streamCompute));
            for (size_t r = 0; r < n_S_rounds; r++) {
                const size_t first = r * (size_t)NR;
                const int    m     = (int)std::min<size_t>((size_t)NR, n_S_chunks - first);
                intkey_t*    buf0  = Sbuf + S_off(r);
                for (int i = 0; i < m; i++) {
                    const size_t n = std::min(chunk_elems, S_el - (first + (size_t)i) * chunk_elems);
                    probe_count_table<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0,
                                            worker->streamCompute>>>(
                        worker->d_key_table, worker->d_cnt_table, ht,
                        buf0 + (size_t)i * chunk_elems, n, worker->g_matches);
                    checkCuda(cudaGetLastError());
                }
                checkCuda(cudaEventRecord(ev_round[r % NRB], worker->streamCompute));
                if (r + 1 < n_S_rounds && (r + 1) >= S_prefetched) {
                    // In parallel mode, S uses a whole-partition buffer and is written once.
                    if (!rs_par && r >= 1) checkCuda(cudaEventSynchronize(ev_round[(r + 1) % NRB]));
                    read_round(S_h, Sbuf + S_off(r + 1), r + 1, n_S_chunks, S_el, "S", &s_ms);
                }
            }
            checkCuda(cudaEventRecord(e_p1, worker->streamCompute));
            wallB_ms = duration<double, std::milli>(steady_clock::now() - tA1).count();
            cuFileHandleDeregister(R_h); close(R_fd);
            cuFileHandleDeregister(S_h); close(S_fd);
        }

        // Finish: wait for kernels, retrieve the count, print timings, and release resources.
        checkCuda(cudaStreamSynchronize(worker->streamCompute));
        ull local = 0;
        checkCuda(cudaMemcpy(&local, worker->g_matches, sizeof(ull), cudaMemcpyDeviceToHost));
        total_result += local;
        float bms = 0, pms = 0;
        cudaEventElapsedTime(&bms, e_b0, e_b1);
        cudaEventElapsedTime(&pms, e_p0, e_p1);
        r_tot += r_ms; s_tot += s_ms;
        double total_meas = duration<double, std::milli>(steady_clock::now() - t_start).count();
        tail_ms = total_meas - setup_ms - wallA_ms - wallB_ms;   // Sync, result copy, deregister, close, and queueing.
        {
            // Aggregate breakdown data. Stage A/B use max() for overlapped read and compute work.
            std::lock_guard<std::mutex> lk(g_js_mu);
            g_js_readR += r_ms;  g_js_readS += s_ms;
            g_js_build += bms;   g_js_probe += pms;
            g_js_setup += setup_ms; g_js_tail += tail_ms;
            if (wallA_ms > 0.0 || wallB_ms > 0.0) {        // Chunked mode: use measured wall-clock stages.
                g_js_stageA += wallA_ms; g_js_stageB += wallB_ms;
            } else {                                        // Whole-partition mode: estimate with max(read, compute).
                g_js_stageA += std::max(r_ms, (double)bms);
                g_js_stageB += std::max(s_ms, (double)pms);
            }
            ++g_js_parts;
        }
        printf("Worker %d: Partition %d completed: result=%llu\n", worker->id, a, local);
        printf("  [pipe] build=%.1f ms probe=%.1f ms | read R=%.1f S=%.1f ms | total=%.1f ms | %s\n",
               bms, pms, r_ms, s_ms,
               duration<double, std::milli>(steady_clock::now() - t_start).count(),
               chunked ? (rs_par ? "chunked + parallel R/S" : "chunked")
                       : (rs_par ? (prefetch ? "whole partition + prefetch + parallel R/S"
                                             : "whole partition + parallel R/S")
                                 : (prefetch ? "whole partition + prefetch"
                                             : "whole partition")));
        if (chunked)
            printf("  [pipe2] Stage wall times: setup=%.1f | A(read R || build)=%.1f | B(read S || probe)=%.1f"
                   " | tail(sync/result/close)=%.1f | sum=%.1f ms (closure error %.1f%%)\n",
                   setup_ms, wallA_ms, wallB_ms, tail_ms,
                   setup_ms + wallA_ms + wallB_ms + tail_ms,
                   100.0 * (setup_ms + wallA_ms + wallB_ms + tail_ms - total_meas) / (total_meas > 0 ? total_meas : 1.0));

        release(a);
        slot = next_slot;
        if (chunked) slot = 0;                       // Chunked mode reuses slot 0.
        if (prefetch) {
            a = b;
            R_ready = (b >= 0);                      // b's R side was prefetched.
            S_ready = (b >= 0) && rs_par;            // S is prefetched only in parallel mode.
        } else {
            a = -1; if (!pick(a)) break;             // Without prefetch, pick the next partition after completion.
            R_ready = false; S_ready = false;        // New partition: read R and S in loop order.
        }
    }

    cudaEventDestroy(e_b0); cudaEventDestroy(e_b1);
    cudaEventDestroy(e_p0); cudaEventDestroy(e_p1);
    cudaEventDestroy(ev_round[0]); cudaEventDestroy(ev_round[1]);
    active_workers--;
}

int GHJ_GDS_JOIN(const std::vector<std::string>& base_paths,
                      int partition_count, int device_id,
                      int num_workers,
                      const std::vector<int>& dir_max_streams) {
    cudaSetDevice(device_id);
    NUM_WORKERS = (num_workers > 0) ? num_workers : 4;
    init_read_config();   // Parse read options before Worker construction and VRAM estimation.

    // Reset breakdown counters per call; one TPC-H query may invoke this function multiple times.
    {
        std::lock_guard<std::mutex> lk(g_js_mu);
        g_js_readR = g_js_readS = g_js_build = g_js_probe = 0;
        g_js_stageA = g_js_stageB = 0;
        g_js_parts = 0;
    }

    g_partition_R_paths.resize(partition_count);
    g_partition_S_paths.resize(partition_count);
    int dir_cnt = (int)base_paths.size();
    for (int i = 0; i < partition_count; i++) {
        int dir_idx = i % dir_cnt;
        g_partition_R_paths[i] = base_paths[dir_idx] + "/file_R/" + std::to_string(i) + ".bin";
        g_partition_S_paths[i] = base_paths[dir_idx] + "/file_S/" + std::to_string(i) + ".bin";
    }
    {   // build the directory lanes (repeated entries collapse by path)
        g_lanes.clear();
        g_partition_lane.assign(partition_count, -1);
        std::map<std::string, int> dir2lane;
        for (int i = 0; i < partition_count; i++) {
            const std::string& d = base_paths[i % dir_cnt];
            auto it = dir2lane.find(d);
            if (it == dir2lane.end()) {
                int dir_idx = i % dir_cnt;
                int lim = (dir_idx < (int)dir_max_streams.size() && dir_max_streams[dir_idx] > 0)
                          ? dir_max_streams[dir_idx] : (num_workers > 0 ? num_workers : 4);
                g_lanes.emplace_back();
                g_lanes.back().limit = lim;
                it = dir2lane.emplace(d, (int)g_lanes.size() - 1).first;
            }
            g_partition_lane[i] = it->second;
        }
        printf("Lane scheduling: %zu directories,", g_lanes.size());
        for (const auto& kv : dir2lane) printf(" [%s limit=%d]", kv.first.c_str(), g_lanes[kv.second].limit);
        printf("\n");
    }

    struct stat st;
    if (stat(base_paths[0].c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Base directory %s does not exist\n", base_paths[0].c_str());
        return -1;
    }

    CHECK_CUFILE(cuFileDriverOpen());

    std::vector<int> valid_partitions;
    g_R_bytes.assign(partition_count, 0);
    g_S_bytes.assign(partition_count, 0);
    size_t max_R_bytes = 0, max_S_bytes = 0;
    for (int i = 0; i < partition_count; i++) {
        if (stat(g_partition_R_paths[i].c_str(), &st) != 0) continue;
        size_t R_bytes = st.st_size;
        if (stat(g_partition_S_paths[i].c_str(), &st) != 0) continue;
        size_t S_bytes = st.st_size;
        if (R_bytes == 0 || S_bytes == 0) continue;
        valid_partitions.push_back(i);
        g_R_bytes[i] = R_bytes;
        g_S_bytes[i] = S_bytes;
        if (R_bytes > max_R_bytes) max_R_bytes = R_bytes;
        if (S_bytes > max_S_bytes) max_S_bytes = S_bytes;
    }

    if (valid_partitions.empty()) {
        fprintf(stderr, "No valid partition files found\n");
        CHECK_CUFILE(cuFileDriverClose());
        return -1;
    }

    total_partitions = (int)valid_partitions.size();
    printf("Max R bytes: %zu, Max S bytes: %zu, Total partitions: %d\n",
           max_R_bytes, max_S_bytes, total_partitions);
    printf("[join] table_size per worker = %zu slots x 8B = %.1f GB\n",
           (max_R_bytes / sizeof(intkey_t)) * HASH_SCALE_FACTOR,
           (max_R_bytes / sizeof(intkey_t)) * HASH_SCALE_FACTOR * 8.0 / 1e9);

    if (NUM_WORKERS > total_partitions) {
        printf("num_workers %d > partitions %d -> clamp to %d\n", NUM_WORKERS, total_partitions, total_partitions);
        NUM_WORKERS = total_partitions;
    }
    {   // VRAM budget: counting tables use the largest R partition; buffers follow the read options.
        size_t max_R_elements = max_R_bytes / sizeof(intkey_t);
        size_t table_bytes = max_R_elements * (size_t)HASH_SCALE_FACTOR * sizeof(int32_t) * 2;
        size_t slot_R, slot_S;
        if (g_chunked) {
            size_t ring = (size_t)(2 * g_n_readers) * g_chunk_bytes;
            slot_R = std::min(ring, max_R_bytes);
            slot_S = g_rs_parallel ? max_S_bytes : std::min(ring, max_S_bytes);
        } else {
            slot_R = max_R_bytes;            // Whole-partition mode: one partition per slot.
            slot_S = max_S_bytes;
        }
        const int n_slots = (g_prefetch_eff || !g_chunked) ? 2 : 1;
        size_t buf_R = slot_R * n_slots;
        size_t buf_S = slot_S * n_slots;
        size_t per_worker = buf_R + buf_S + table_bytes;
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            int fit = (int)((size_t)(free_b * 0.85) / (per_worker > 0 ? per_worker : 1));
            printf("[mem] per-worker=%.2f GB (tables %.2f GB, R/S buffers %.2f GB, slots %d%s%s), free=%.2f GB -> max %d workers\n",
                   per_worker / 1e9, table_bytes / 1e9, (buf_R + buf_S) / 1e9, n_slots,
                   g_chunked ? ", chunked" : ", whole partition",
                   g_rs_parallel ? ", parallel R/S (whole S)" : "", free_b / 1e9, fit);
            if (fit < NUM_WORKERS) {
                printf("num_workers %d -> %d (VRAM budget limit)\n", NUM_WORKERS, fit > 0 ? fit : 1);
                NUM_WORKERS = (fit > 0) ? fit : 1;
            }
        }
    }
    for (DirLane& l : g_lanes) { l.pending.clear(); l.active = 0; }
    total_result         = 0;
    completed_partitions = 0;
    all_done             = false;
    active_workers       = 0;

    // Initialize workers in parallel.
    auto t_worker_init = std::chrono::steady_clock::now();
    printf("Creating %d workers (parallel initialization)...\n", NUM_WORKERS);
    std::vector<Worker*> worker_ptrs(NUM_WORKERS, nullptr);
    {
        std::vector<std::thread> init_threads;
        init_threads.reserve(NUM_WORKERS);
        for (int i = 0; i < NUM_WORKERS; i++) {
            init_threads.emplace_back([&worker_ptrs, i, device_id, max_R_bytes, max_S_bytes]() {
                worker_ptrs[i] = new Worker(i, device_id, max_R_bytes, max_S_bytes);
            });
        }
        for (auto& t : init_threads) t.join();
    }
    double init_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_worker_init).count();
    printf("All workers created (parallel init: %.1f ms). Starting threads...\n", init_ms);

    std::vector<std::thread> workers;
    workers.reserve(NUM_WORKERS);
    auto t_join0 = std::chrono::steady_clock::now();   // Join-phase wall-clock start.
    for (Worker* w : worker_ptrs)
        workers.emplace_back(worker_function, w);

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (int pid : valid_partitions) g_lanes[g_partition_lane[pid]].pending.push_back(pid);
    }
    cv.notify_all();

    while (completed_partitions.load() < total_partitions)
        std::this_thread::yield();

    {
        // Must be set while holding queue_mutex: the worker predicate check and its condvar registration have to be mutually exclusive with it,
        // otherwise a worker that judged the predicate false and was not yet registered misses the notify - the lost wakeup behind the hangs (2026-09-25)
        std::lock_guard<std::mutex> lock(queue_mutex);
        all_done = true;
    }
    cv.notify_all();
    for (auto& t : workers) t.join();
    for (Worker* w : worker_ptrs) delete w;

    printf("All partitions completed. Total result: %llu\n", total_result.load());
    {
        // Breakdown data. Use non-overlapping stageA and stageB for plotting; their
        // sum approximates join_wall. readR/readS/build/probe represent work and
        // must not be stacked as elapsed time.
        double join_wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_join0).count();
        double work = g_js_readR + g_js_readS + g_js_build + g_js_probe;
        printf("[join-summary] parts=%d | work: readR=%.1f readS=%.1f build=%.1f probe=%.1f ms (sum=%.1f)\n",
               g_js_parts, g_js_readR, g_js_readS, g_js_build, g_js_probe, work);
        printf("[join-summary] stages (sum over partitions): setup=%.1f  stageA=%.1f  stageB=%.1f"
               "  tail=%.1f ms | join_wall=%.1f ms | overlap benefit=%.1f ms (%.0f%% of work)\n",
               g_js_setup, g_js_stageA, g_js_stageB, g_js_tail, join_wall, work - join_wall,
               work > 0 ? 100.0 * (work - join_wall) / work : 0.0);
    }

    CHECK_CUFILE(cuFileDriverClose());
    return 0;
}
