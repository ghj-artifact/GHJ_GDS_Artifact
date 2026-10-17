/*
gpu_hash_join.cuh - joiner interface (counting hash table + per-directory read lanes)

Included by GHJ_GDS_WPS and GHJ_GDS_SMH. Public entry point:
int GHJ_GDS_JOIN(base_paths, partition_count, device_id, num_workers = 4, const std::vector<int>& dir_max_streams = {});

The chunked pipeline, cross-partition prefetching, and parallel R/S reads are
runtime options.
*/
#ifndef GPU_HASH_JOIN_H
#define GPU_HASH_JOIN_H

#include <string>
#include <vector>

// The counting hash table avoids the high-Zipf skew problem of chained probing.
// Directory lanes bound concurrent reads without changing the join interface.
int GHJ_GDS_JOIN(const std::vector<std::string>& base_paths,
                   int partition_count, int device_id,
                   int num_workers = 4,
                   const std::vector<int>& dir_max_streams = {});

#endif
