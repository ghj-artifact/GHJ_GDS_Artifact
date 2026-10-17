/*
GHJ_GDS_WPS - chunked-pipeline join with a counting hash table, configurable
read structure, and a 32-slot TPC-H spill layout.

Build:
nvcc -std=c++17 -O3 -Xcompiler -fopenmp \
    ../Data_Generate/gpu_data_generator.cu \
    ./Partitioner/gpu_partitioner_wps.cu \
    ./Joiner/gpu_hash_join.cu \
    GHJ_GDS_WPS.cu \
    -o GHJ_GDS_WPS \
    -lcufile

Usage:
  ./GHJ_GDS_WPS [partition_count] [R] [S] [zipf_alpha] [target_batches] [num_workers]   # random / Zipf
  ./GHJ_GDS_WPS --tpch <Q2|Q3|Q4|Q5> <SF> [join_name|all] [P] [batches] [workers]        # TPC-H
  If an unrelated process holds a GPU in exclusive mode:
  CUDA_VISIBLE_DEVICES=1,2 GHJ_DEVICEID=1 ./GHJ_GDS_WPS ...
  GHJ_LAND="<dir>[:<slots>[:<lanes>]][,...]"   # Override the spill layout.

Implementation:
  partitioner: gpu_partitioner_wps.cu uses warp-shuffle prefix sums in
    compute_partitions_and_offsets_wps.
  joiner: gpu_hash_join.cu uses a counting hash table and a read pipeline.
    Stage A reads R chunks while building; stage B reads S chunks while probing.
    Every probe starts after all build work completes. Read options are selected
    through environment variables and default to chunked reads.
  This file configures four-disk spill distribution, read lanes, and the
    32-slot TPC-H layout.

The chunked pipeline hides build work behind R reads and overlaps probe work
with subsequent S reads. Defaults are 128 MiB chunks, one reader per round, and
prefetching disabled. Override them with GHJ_CHUNK_MB, GHJ_CHUNK_READERS,
GHJ_PREFETCH, and GHJ_RS_PARALLEL.

Test requirements:
  - For a single-GPU run, select the intended GPU and set
    CUDA_MPS_PIPE_DIRECTORY to an empty directory owned by the run.
  - Use 4 MiB cuFile I/O chunks:
    export CUFILE_ENV_PATH_JSON=<artifact-root>/GHJ_GDS/cufile_gds4m.json
  - Report the program's preprocess, join, and total timings. Input loading is
    outside the reported metric.
*/

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufile.h>
#include "../Data_Generate/gpu_data_generator.cuh"
#include "./Partitioner/gpu_partitioner_wps.cuh"
#include "./Joiner/gpu_hash_join.cuh"
#include "../Data_Generate/tpch_key_loader.cuh"
#include "types_GDS.h"

// CUDA device ordinal. Override with GHJ_DEVICEID.
// CUDA ordinals and nvidia-smi indices may differ; always verify the device map
// on the target host.
// Set CUDA_MPS_PIPE_DIRECTORY to an isolated directory when a shared MPS daemon
// is present. cuFile operations may fail if any visible GPU is held in exclusive
// process mode.
static int DEVICEID = 2;

// Default spill layout for random and Zipf workloads: 12 slots distributed
// evenly across four directories, with four read lanes per directory.
// GHJ_LAND overrides this layout. Partition i maps to slot i % slot_count.
std::vector<std::string> BASE_PATHS = {
    "/mnt/nvme3/tempfile",                  // A
    "/var/tempfile",                        // B
    "/mnt/nvme2/tmp/ghj_gds_tempfile",      // C
    "/mnt/nvme1/tempfile/GDS_test",         // D
    "/mnt/nvme3/tempfile",                  // A
    "/var/tempfile",                        // B
    "/mnt/nvme2/tmp/ghj_gds_tempfile",      // C
    "/mnt/nvme1/tempfile/GDS_test",         // D
    "/mnt/nvme3/tempfile",                  // A
    "/var/tempfile",                        // B
    "/mnt/nvme2/tmp/ghj_gds_tempfile",      // C
    "/mnt/nvme1/tempfile/GDS_test"          // D
};

// Cap on concurrent in-flight reads per spill directory (1:1 with BASE_PATHS; equal paths collapse).
// Read-lane counts have little effect in the measured configuration.
std::vector<int> DIR_MAX_STREAMS = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

// Parse GHJ_LAND into BASE_PATHS and DIR_MAX_STREAMS.
static void init_land_layout() {
    const char* spec = getenv("GHJ_LAND");
    if (!spec || !*spec) {
        printf("[land] GHJ_LAND is unset; using the default 12-slot layout with four lanes per directory\n");
        return;
    }
    std::vector<std::string> paths;
    std::vector<int> lanes;
    std::string s(spec);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, (comma == std::string::npos ? s.size() : comma) - pos);
        pos = (comma == std::string::npos) ? s.size() + 1 : comma + 1;
        if (tok.empty()) continue;
        std::vector<std::string> parts;
        size_t p0 = 0;
        while (true) {
            size_t c = tok.find(':', p0);
            parts.push_back(tok.substr(p0, (c == std::string::npos ? tok.size() : c) - p0));
            if (c == std::string::npos) break;
            p0 = c + 1;
        }
        int slots = (parts.size() > 1 && !parts[1].empty()) ? atoi(parts[1].c_str()) : 1;
        int lane  = (parts.size() > 2 && !parts[2].empty()) ? atoi(parts[2].c_str()) : 4;
        if (slots < 1) slots = 1;
        if (lane  < 1) lane  = 1;
        for (int k = 0; k < slots; k++) { paths.push_back(parts[0]); lanes.push_back(lane); }
    }
    if (paths.empty()) {
        printf("[land] GHJ_LAND parsed to an empty layout; using the default layout\n");
        return;
    }
    BASE_PATHS.swap(paths);
    DIR_MAX_STREAMS.swap(lanes);
    printf("[land] GHJ_LAND active: %zu slots ->", BASE_PATHS.size());
    for (size_t i = 0; i < BASE_PATHS.size(); i++)
        printf("  %s(lanes=%d)", BASE_PATHS[i].c_str(), DIR_MAX_STREAMS[i]);
    printf("\n");
}

// Configure 4 MiB cuFile I/O chunks for this process through
// CUFILE_ENV_PATH_JSON. Call before cuFileDriverOpen().
// GHJ_CUFILE_4M toggles the setting; GHJ_CUFILE_JSON overrides the file path.
static void init_cufile_tuning(bool default_on) {
    const char* off = getenv("GHJ_CUFILE_4M");
    bool on = off ? (atoi(off) != 0) : default_on;
    if (!on) { printf("[cufile] 4 MiB chunk tuning disabled (GHJ_CUFILE_4M=0)\n"); return; }
    if (getenv("CUFILE_ENV_PATH_JSON")) {
        printf("[cufile] Keeping external CUFILE_ENV_PATH_JSON=%s\n", getenv("CUFILE_ENV_PATH_JSON"));
        return;
    }
    const char* env_p = getenv("GHJ_CUFILE_JSON");
    std::vector<std::string> cand;
    if (env_p) cand.push_back(env_p);
    cand.push_back("cufile_gds4m.json");
    for (const auto& p : cand) {
        if (access(p.c_str(), R_OK) == 0) {
            setenv("CUFILE_ENV_PATH_JSON", p.c_str(), 0);
            printf("[cufile] 4 MiB chunk tuning enabled: CUFILE_ENV_PATH_JSON=%s (max_direct_io_size_kb=4096)\n", p.c_str());
            return;
        }
    }
    printf("[cufile] cufile_gds4m.json not found; using the system cuFile configuration\n");
}


// -- Spill layout for TPC-H mode (32 slots, A:B:C = 13:13:6 = 40.625%:40.625%:18.75%) --
const std::vector<std::string> BASE_PATHS_TPCH = {
    "/mnt/nvme3/tempfile",                 // slot 0: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 1: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 2: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 3: B  nvme2n1  NUMA0  local
    "/mnt/nvme2/tmp/ghj_gds_tempfile",     // slot 4: C  nvme1n1  NUMA1  remote
    "/mnt/nvme3/tempfile",                 // slot 5: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 6: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 7: A  nvme3n1  NUMA0  local
    "/mnt/nvme2/tmp/ghj_gds_tempfile",     // slot 8: C  nvme1n1  NUMA1  remote
    "/var/tempfile",                       // slot 9: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 10: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 11: B  nvme2n1  NUMA0  local
    "/mnt/nvme2/tmp/ghj_gds_tempfile",     // slot 12: C  nvme1n1  NUMA1  remote
    "/mnt/nvme3/tempfile",                 // slot 13: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 14: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 15: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 16: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 17: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 18: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 19: A  nvme3n1  NUMA0  local
    "/mnt/nvme2/tmp/ghj_gds_tempfile",     // slot 20: C  nvme1n1  NUMA1  remote
    "/var/tempfile",                       // slot 21: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 22: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 23: B  nvme2n1  NUMA0  local
    "/mnt/nvme2/tmp/ghj_gds_tempfile",     // slot 24: C  nvme1n1  NUMA1  remote
    "/mnt/nvme3/tempfile",                 // slot 25: A  nvme3n1  NUMA0  local
    "/var/tempfile",                       // slot 26: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 27: A  nvme3n1  NUMA0  local
    "/mnt/nvme2/tmp/ghj_gds_tempfile",     // slot 28: C  nvme1n1  NUMA1  remote
    "/var/tempfile",                       // slot 29: B  nvme2n1  NUMA0  local
    "/mnt/nvme3/tempfile",                 // slot 30: A  nvme3n1  NUMA0  local
    "/var/tempfile"                        // slot 31: B  nvme2n1  NUMA0  local
};

// Cap on concurrent in-flight reads per spill directory (1:1 with BASE_PATHS; C, the remote slow disk, allows 1)
const std::vector<int> DIR_MAX_STREAMS_TPCH = {4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4};

std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_vec;

int solve_massive(int partition_count, long long data_size_R, long long data_size_S,
                  double zipf_alpha,
                  const std::vector<std::string>& base_paths, size_t target_batches,
                  int num_workers)
{
    GPURandomDataGen::GPUIntGenerator gpu_gen(DEVICEID);

    std::cout << "  - generating R keys (pinned, n=" << data_size_R
              << ", zipf_alpha=" << zipf_alpha << ")...\n";
    // Key buffers use int32_t (matching the generator template argument T): preprocess_gpu_gds() takes const int*,
    // and the cache name is derived from signed int32 semantics (min -2147483648 / max 2147483647).
    GPURandomDataGen::PinnedBuffer<int32_t> gpu_pk, gpu_fk;
    if (zipf_alpha > 0.0) {
        // Key domain follows the larger relation so that the R and S key ranges overlap
        // For sizes above INT32_MAX the upper bound is clamped to INT32_MAX (same as multi-gpu-sort-merge-join)
        int32_t lo = 0;
        long long raw_hi = std::max(data_size_R, data_size_S) - 1;
        int32_t hi = (int32_t)std::min<long long>(
            raw_hi, (long long)std::numeric_limits<int32_t>::max());
        gpu_pk = gpu_gen.random_ints_zipf_pinned(data_size_R, zipf_alpha, lo, hi, 42);
        std::cout << "  - generating S keys (pinned, n=" << data_size_S << ")...\n";
        gpu_fk = gpu_gen.random_ints_zipf_pinned(data_size_S, zipf_alpha, lo, hi, 43);
    } else {
        gpu_pk = gpu_gen.random_ints_pinned(data_size_R);
        std::cout << "  - generating S keys (pinned, n=" << data_size_S << ")...\n";
        gpu_fk = gpu_gen.random_ints_pinned(data_size_S, std::numeric_limits<int32_t>::min(),
                                             std::numeric_limits<int32_t>::max(), 43);
    }

    printf("size: %zu %zu\n", gpu_pk.size(), gpu_fk.size());

    time_vec.push_back(std::chrono::high_resolution_clock::now());

    long long max_data = std::max(data_size_R, data_size_S);
    size_t max_batch = (size_t)(max_data + target_batches - 1) / target_batches;

    GPUPartitioner gpu_partitioner(partition_count, DEVICEID, base_paths, max_batch);
    gpu_partitioner.preprocess_gpu_gds(true, gpu_pk.data(), gpu_pk.size());
    gpu_partitioner.preprocess_gpu_gds(false, gpu_fk.data(), gpu_fk.size());

    std::cout << "preprocess finished\n";

    time_vec.push_back(std::chrono::high_resolution_clock::now());
    auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_vec[1] - time_vec[0]);
    std::cout << "preprocess time: " << preprocess_time.count() << " ms\n";

    GHJ_GDS_JOIN(base_paths, partition_count, DEVICEID, num_workers, DIR_MAX_STREAMS);

    time_vec.push_back(std::chrono::high_resolution_clock::now());
    auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(time_vec[2] - time_vec[0]);
    auto GPUonly   = std::chrono::duration_cast<std::chrono::milliseconds>(time_vec[2] - time_vec[1]);
    std::cout << "total time:      " << duration.count()        << " ms\n";
    std::cout << "preprocess time: " << preprocess_time.count() << " ms\n";
    std::cout << "Hash Join time:  " << GPUonly.count()         << " ms\n";

    gpu_pk = GPURandomDataGen::PinnedBuffer<int32_t>();
    gpu_fk = GPURandomDataGen::PinnedBuffer<int32_t>();
    return 0;
}

// New mode: TPC-H cached join key benchmark

struct TpchJoinTiming {
    std::string join_name;
    size_t build_count;
    size_t probe_count;
    long long load_ms;
    long long preprocess_ms;
    long long join_ms;
    long long total_ms;
};

int run_tpch_join(const std::string& query, int sf, const std::string& join_name,
                  int partition_count,
                  const std::vector<std::string>& base_paths, size_t target_batches,
                  int num_workers,
                  TpchJoinTiming& out_timing)
{
    out_timing.join_name = join_name;

    // ── 1. Load keys from cache ────────────────────────────────────────────
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "TPC-H Join: " << join_name << "  (Query=" << query
              << ", SF=" << sf << ")\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    auto t_load_start = std::chrono::high_resolution_clock::now();

    TpchJoinKeys join_keys;
    try {
        join_keys = load_tpch_join_keys(query, sf, join_name, true);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to load keys for " << join_name
                  << ": " << e.what() << std::endl;
        return 1;
    }

    auto t_load_end = std::chrono::high_resolution_clock::now();
    out_timing.load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_load_end - t_load_start).count();
    out_timing.build_count = join_keys.build_count;
    out_timing.probe_count = join_keys.probe_count;

    std::cout << "  Load time: " << out_timing.load_ms << " ms\n";

    if (join_keys.build_count == 0 && join_keys.probe_count == 0) {
        std::cout << "  SKIP: both sides empty; no join to run.\n";
        out_timing.preprocess_ms = 0;
        out_timing.join_ms = 0;
        out_timing.total_ms = out_timing.load_ms;
        return 0;
    }

    // ── 2. Partition + Hash Join ──────────────────────────────────────────
    size_t max_data = std::max(join_keys.build_count, join_keys.probe_count);
    size_t max_batch = (max_data + target_batches - 1) / target_batches;

    std::cout << "  Partition count: " << partition_count
              << ", target_batches: " << target_batches
              << ", max_batch: " << max_batch << "\n";

    auto t_part_start = std::chrono::high_resolution_clock::now();

    GPUPartitioner gpu_partitioner(partition_count, DEVICEID, base_paths, max_batch);
    gpu_partitioner.preprocess_gpu_gds(true,
        reinterpret_cast<const int*>(join_keys.build_keys.data()), join_keys.build_keys.size());
    gpu_partitioner.preprocess_gpu_gds(false,
        reinterpret_cast<const int*>(join_keys.probe_keys.data()), join_keys.probe_keys.size());

    std::cout << "  partition finished\n";

    auto t_part_end = std::chrono::high_resolution_clock::now();
    out_timing.preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_part_end - t_part_start).count();

    GHJ_GDS_JOIN(base_paths, partition_count, DEVICEID, num_workers, DIR_MAX_STREAMS_TPCH);

    auto t_join_end = std::chrono::high_resolution_clock::now();
    out_timing.join_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_join_end - t_part_end).count();
    out_timing.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_join_end - t_load_start).count();

    std::cout << "  ─────────────────────────────────────\n";
    std::cout << "  preprocess time: " << out_timing.preprocess_ms << " ms\n";
    std::cout << "  Hash Join time:  " << out_timing.join_ms << " ms\n";
    std::cout << "  total time:      " << out_timing.total_ms << " ms"
              << "  (incl. load " << out_timing.load_ms << " ms)\n";

    return 0;
}


int solve_tpch_benchmark(const std::string& query, int sf,
                          const std::string& join_name_filter,
                          int partition_count,
                          const std::vector<std::string>& base_paths,
                          size_t target_batches,
                          int num_workers)
{
    // ── Determine which joins to run ───────────────────────────────────────
    const std::vector<std::string>* join_names = nullptr;
    if (query == "Q2") {
        join_names = &Q2_JOIN_NAMES;
    } else if (query == "Q3") {
        join_names = &Q3_JOIN_NAMES;
    } else if (query == "Q4") {
        join_names = &Q4_JOIN_NAMES;
    } else if (query == "Q5") {
        join_names = &Q5_JOIN_NAMES;
    } else {
        std::cerr << "ERROR: Unknown query '" << query
                  << "'. Supported: Q2, Q3, Q4, Q5.\n";
        return 1;
    }

    std::vector<std::string> selected_joins;
    if (join_name_filter.empty() || join_name_filter == "all") {
        selected_joins = *join_names;
    } else {
        // Try exact match first, then auto-prepend query prefix
        // e.g. "join2_orders_lineitem" -> "Q5_join2_orders_lineitem"
        std::string resolved = join_name_filter;
        bool found = false;
        for (const auto& name : *join_names) {
            if (name == resolved) { found = true; break; }
        }
        if (!found) {
            std::string prefixed = query + "_" + join_name_filter;
            for (const auto& name : *join_names) {
                if (name == prefixed) { resolved = prefixed; found = true; break; }
            }
        }
        if (!found) {
            std::cerr << "ERROR: Unknown join '" << join_name_filter
                      << "' for " << query << ".\n";
            std::cerr << "  Available joins:";
            for (const auto& name : *join_names)
                std::cerr << " " << name;
            std::cerr << "\n";
            return 1;
        }
        selected_joins.push_back(resolved);
    }

    // ── cuFileDriverOpen outside timing (same policy as old mode) ─────────
    auto t_open = std::chrono::high_resolution_clock::now();
    CUfileError_t open_st = cuFileDriverOpen();
    if (open_st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileDriverOpen failed: %d\n", (int)open_st.err);
        return 1;
    }
    double open_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_open).count();
    printf("cuFileDriverOpen (pre-init, outside timing): %.1f ms\n", open_ms);

    // ── Run each join ─────────────────────────────────────────────────────
    std::vector<TpchJoinTiming> all_timings;
    int overall_rc = 0;

    for (const auto& join_name : selected_joins) {
        TpchJoinTiming timing;
        int rc = run_tpch_join(query, sf, join_name, partition_count,
                                base_paths, target_batches, num_workers, timing);
        all_timings.push_back(timing);
        if (rc != 0) overall_rc = rc;
    }

    cuFileDriverClose();

    // ── Summary ───────────────────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TPC-H " << query << "  SF=" << sf
              << "  Benchmark Summary";
    for (int pad = 56 - (int)query.size(); pad > 0; pad--)
        std::cout << " ";
    std::cout << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  "
              << std::left  << std::setw(36) << "Join"
              << std::right << std::setw(10) << "Build"
              << std::setw(10) << "Probe"
              << std::setw(10) << "Load(ms)"
              << std::setw(10) << "Part(ms)"
              << std::setw(10) << "Join(ms)"
              << std::setw(10) << "Total(ms)"
              << "  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

    long long sum_preprocess = 0, sum_join = 0, sum_total = 0;
    for (const auto& t : all_timings) {
        std::cout << "║  "
                  << std::left  << std::setw(36) << t.join_name
                  << std::right << std::setw(10) << t.build_count
                  << std::setw(10) << t.probe_count
                  << std::setw(10) << t.load_ms
                  << std::setw(10) << t.preprocess_ms
                  << std::setw(10) << t.join_ms
                  << std::setw(10) << t.total_ms
                  << "  ║\n";
        sum_preprocess += t.preprocess_ms;
        sum_join       += t.join_ms;
        sum_total      += t.total_ms;
    }
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  "
              << std::left  << std::setw(36) << "SUM"
              << std::right << std::setw(10) << ""
              << std::setw(10) << ""
              << std::setw(10) << ""
              << std::setw(10) << sum_preprocess
              << std::setw(10) << sum_join
              << std::setw(10) << sum_total
              << "  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

    return overall_rc;
}


// Usage
void print_usage(const char* prog)
{
    std::cerr << "Usage:\n";
    std::cerr << "  Random / Zipf mode: " << prog
              << " [partition_count] [data_size_R] [data_size_S] [zipf_alpha] [target_batches] [num_workers]\n";
    std::cerr << "  TPC-H mode:         " << prog
              << " --tpch <Q2|Q3|Q4|Q5> <SF> [join_name|all] [partition_count] [target_batches] [num_workers]\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  " << prog << " 4 1000000000 1000000000 0.0 12 4\n";
    std::cerr << "  " << prog << " --tpch Q2 100 all 16 12 4\n";
    std::cerr << "  " << prog << " --tpch Q5 400 join4_customer_supplier_nation 16 12 4\n";
}


int main(int argc, char* argv[])
{
    // -- DEVICEID override (default 2 = H100; see the file header) ------------------
    if (const char* e = getenv("GHJ_DEVICEID")) {
        DEVICEID = std::atoi(e);
        std::cout << "[env] GHJ_DEVICEID=" << DEVICEID
                  << " (CUDA_VISIBLE_DEVICES=" << (getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "unset") << ")\n";
    }

    // Configure cuFile before mode dispatch because the TPC-H branch returns early.
    init_cufile_tuning(true);

    // -- Mode dispatch: "--tpch" as first argument runs the TPC-H cache-key benchmark; otherwise the random/zipf one --
    if (argc > 1 && std::strcmp(argv[1], "--tpch") == 0) {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        std::string query        = argv[2];
        int         sf           = std::stoi(argv[3]);
        std::string join_filter  = (argc > 4) ? argv[4] : "all";
        int         pc           = (argc > 5) ? std::stoi(argv[5]) : 4;
        size_t      tb           = (argc > 6) ? std::stoull(argv[6]) : 12;
        int         nw           = (argc > 7) ? std::stoi(argv[7]) : 4;
        if (sf <= 0 || pc <= 0 || (long long)tb <= 0) {
            std::cerr << "Error: SF / partition_count / target_batches must be positive.\n";
            return 1;
        }
        std::cout << "Running GHJ_GDS_WPS TPC-H Benchmark Mode\n"
                  << "  query=" << query << "  SF=" << sf
                  << "  join_filter=" << join_filter
                  << "  partition_count=" << pc
                  << "  target_batches=" << tb
                  << "  num_workers=" << nw << "\n"
                  << "  DEVICEID=" << DEVICEID << "\n"
                  << "  BASE_PATHS_TPCH: ";
        for (const auto& p : BASE_PATHS_TPCH) std::cout << p << "  ";
        std::cout << "\n  DIR_MAX_STREAMS_TPCH: ";
        for (int v : DIR_MAX_STREAMS_TPCH) std::cout << v << " ";
        std::cout << "\n";
        return solve_tpch_benchmark(query, sf, join_filter, pc, BASE_PATHS_TPCH, tb, nw);
    }

    int partition_count      = 4;
    long long data_size_R    = 1000000000;
    long long data_size_S    = 1000000000;
    double zipf_alpha        = 0.0;
    size_t target_batches    = 12;
    int    num_workers       = 4;

    if (argc > 1) partition_count = std::stoi(argv[1]);
    if (argc > 2) data_size_R     = std::stoll(argv[2]);
    if (argc > 3) data_size_S     = std::stoll(argv[3]);
    if (argc > 4) zipf_alpha      = std::stod(argv[4]);
    if (argc > 5) target_batches  = std::stoull(argv[5]);
    if (argc > 6) num_workers     = std::stoi(argv[6]);

    if (partition_count <= 0 || data_size_R <= 0 || data_size_S <= 0 || (long long)target_batches <= 0) {
        std::cerr << "Error: partition_count, data_size_R, data_size_S, target_batches must be positive.\n";
        return 1;
    }

    init_land_layout();   // Apply GHJ_LAND, or keep the default four-directory layout.

    std::cout << "Running GHJ_GDS_WPS (12-slot spill layout, read lanes, and chunked-pipeline counting joiner)\n"
              << "  partition_count=" << partition_count
              << "  data_size_R=" << data_size_R
              << "  data_size_S=" << data_size_S
              << "  zipf_alpha=" << zipf_alpha
              << "  target_batches=" << target_batches << "\n"
              << "  num_workers=" << num_workers << "\n"
              << "  BASE_PATHS: ";
    for (const auto& p : BASE_PATHS) std::cout << p << "  ";
    std::cout << "\n";

    // Default read structure. User-provided values take precedence.
    // The chunked pipeline uses 128 MiB chunks and disables cross-partition prefetch.
    if (!getenv("GHJ_CHUNK_MB"))    setenv("GHJ_CHUNK_MB", "128", 0);
    if (!getenv("GHJ_PREFETCH")) setenv("GHJ_PREFETCH", "0", 0);

    // cuFileDriverOpen initialises the GDS driver outside the timed region
    auto t_open = std::chrono::high_resolution_clock::now();
    CUfileError_t open_st = cuFileDriverOpen();
    if (open_st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileDriverOpen failed: %d\n", (int)open_st.err);
        return 1;
    }
    double open_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_open).count();
    printf("cuFileDriverOpen (pre-init, outside timing): %.1f ms\n", open_ms);

    solve_massive(partition_count, data_size_R, data_size_S, zipf_alpha, BASE_PATHS,
                  target_batches, num_workers);

    cuFileDriverClose();
    return 0;
}
