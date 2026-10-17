#pragma once
/*
tpch_key_loader.cuh - TPC-H join-key cache loader

  load_tpch_join_keys(query, sf, join_name, pinned) reads
  tpch_{query}_SF{sf}_{join_name}_{R|S}.bin from the cache directory.
  The binary format is [8-byte element count][uint32_t elements].
  The per-query join-name tables (Q2=4, Q3=2, Q4=1, Q5=5) are also defined here.
*/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <filesystem>
#include <sstream>

#include "../Data_Generate/gpu_data_generator.cuh"  // PinnedBuffer definition.

// Cache-path construction; must match the key-extraction script.

inline std::string get_tpch_cache_name(const std::string& query,
                                        int sf,
                                        const std::string& join_name,
                                        const std::string& side)
{
    std::ostringstream oss;
    oss << "tpch_" << query << "_SF" << sf << "_" << join_name << "_" << side << ".bin";
    return oss.str();
}

inline std::filesystem::path get_tpch_cache_path(const std::string& query,
                                                   int sf,
                                                   const std::string& join_name,
                                                   const std::string& side)
{
    // Keep this cache directory consistent with get_cache_path().
    static const std::filesystem::path cache_dir = "/mnt/sdc/srv/test_data_GDS";
    return cache_dir / get_tpch_cache_name(query, sf, join_name, side);
}


// Load a uint32_t key array from a binary file into pinned host memory.
inline GPURandomDataGen::PinnedBuffer<uint32_t> load_keys_pinned(
    const std::string& filepath,
    bool verbose = true)
{
    namespace fs = std::filesystem;

    if (!fs::exists(filepath)) {
        throw std::runtime_error("File not found: " + filepath);
    }

    auto file_size = fs::file_size(filepath);
    if (file_size < 8) {
        throw std::runtime_error("File too short (no header): " + filepath);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Open the input file.
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    // Read the element count (8-byte little-endian size_t).
    size_t n = 0;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n == 0) {
        if (verbose) {
            std::cout << "Loaded (pinned) from " << filepath
                      << ": 0 keys (empty file)" << std::endl;
        }
        return GPURandomDataGen::PinnedBuffer<uint32_t>(0);
    }

    // Validate the file size.
    size_t expected_size = sizeof(size_t) + n * sizeof(uint32_t);
    if (file_size < expected_size) {
        throw std::runtime_error(
            "File size mismatch in " + filepath +
            ": expected >= " + std::to_string(expected_size) +
            " bytes, got " + std::to_string(file_size));
    }

    // Allocate the pinned buffer and read the key data.
    GPURandomDataGen::PinnedBuffer<uint32_t> buf(n);
    ifs.read(reinterpret_cast<char*>(buf.data()), n * sizeof(uint32_t));

    if (ifs.fail() && !ifs.eof()) {
        throw std::runtime_error("Failed to read key data from: " + filepath);
    }

    ifs.close();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    if (verbose) {
        std::cout << "Loaded (pinned) from " << filepath << ": "
                  << n << " keys in " << ms << " ms ("
                  << (file_size / 1024.0 / 1024.0) << " MB)" << std::endl;
    }

    return buf;
}


// Load one TPC-H key file from the cache.
inline GPURandomDataGen::PinnedBuffer<uint32_t> load_tpch_keys_pinned(
    const std::string& query,
    int sf,
    const std::string& join_name,
    const std::string& side,
    bool verbose = true)
{
    auto path = get_tpch_cache_path(query, sf, join_name, side);
    return load_keys_pinned(path.string(), verbose);
}


// A build/probe join-key pair with metadata.
struct TpchJoinKeys {
    std::string name;
    GPURandomDataGen::PinnedBuffer<uint32_t> build_keys;  // R side (PK)
    GPURandomDataGen::PinnedBuffer<uint32_t> probe_keys;  // S side (FK)
    size_t build_count;
    size_t probe_count;
};


// Load a build/probe join-key pair from the cache.
inline TpchJoinKeys load_tpch_join_keys(
    const std::string& query,
    int sf,
    const std::string& join_name,
    bool verbose = true)
{
    TpchJoinKeys result;
    result.name = join_name;

    auto r_path = get_tpch_cache_path(query, sf, join_name, "R");
    auto s_path = get_tpch_cache_path(query, sf, join_name, "S");

    if (verbose) {
        std::cout << "\nLoading TPC-H join key pair: " << join_name
                  << " (" << query << ", SF=" << sf << ")" << std::endl;
        std::cout << "  R: " << r_path.string() << std::endl;
        std::cout << "  S: " << s_path.string() << std::endl;
    }

    result.build_keys = load_keys_pinned(r_path.string(), verbose);
    result.build_count = result.build_keys.size();

    result.probe_keys = load_keys_pinned(s_path.string(), verbose);
    result.probe_count = result.probe_keys.size();

    if (verbose) {
        std::cout << "  Join pair loaded: Build (R)=" << result.build_count
                  << " keys, Probe (S)=" << result.probe_count
                  << " keys" << std::endl;
    }

    return result;
}


// Load a join-key pair from an explicitly specified directory.
inline TpchJoinKeys load_join_keys(
    const std::string& name,
    const std::string& keys_dir,
    bool verbose = true)
{
    TpchJoinKeys result;
    result.name = name;

    std::string r_path = keys_dir + "/" + name + "_R.dat";
    std::string s_path = keys_dir + "/" + name + "_S.dat";

    if (verbose) {
        std::cout << "\nLoading join key pair: " << name << std::endl;
    }

    result.build_keys = load_keys_pinned(r_path, verbose);
    result.build_count = result.build_keys.size();

    result.probe_keys = load_keys_pinned(s_path, verbose);
    result.probe_count = result.probe_keys.size();

    if (verbose) {
        std::cout << "  Build (R): " << result.build_count << " keys"
                  << ", Probe (S): " << result.probe_count << " keys"
                  << std::endl;
    }

    return result;
}


// Predefined join-name lists for the TPC-H queries.

// Q2 has four joins. Names match the key-extraction script and include the query prefix.
const std::vector<std::string> Q2_JOIN_NAMES = {
    "Q2_join1_nation_region",
    "Q2_join2_supplier_nation",
    "Q2_join3_part_partsupp",
    "Q2_join4_supplier_partsupp",
};

// Q3 has two joins.
const std::vector<std::string> Q3_JOIN_NAMES = {
    "Q3_join1_customer_orders",
    "Q3_join2_orders_lineitem",
};

// Q4 has one join.
//   Q4_join1: orders.o_orderkey = lineitem.l_orderkey
const std::vector<std::string> Q4_JOIN_NAMES = {
    "Q4_join1_orders_lineitem",
};

// Q5 has five joins.
const std::vector<std::string> Q5_JOIN_NAMES = {
    "Q5_join1_customer_orders",
    "Q5_join2_orders_lineitem",
    "Q5_join3_lineitem_supplier",
    "Q5_join4_customer_supplier_nation",
    "Q5_join5_nation_region",
};
