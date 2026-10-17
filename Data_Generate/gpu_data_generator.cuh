/*
gpu_data_generator.cuh - random data generation interface.
Generates uniform and Zipf-distributed integer keys for the main programs.
*/
#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <chrono>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <type_traits>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <functional>
#include <numeric>
#include <cstring>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>  // Used only for foreign-key parameter types.

namespace GPURandomDataGen
{
    // Hashing and cache-path helpers.
    template<typename... Args>
    size_t hash_combine(size_t seed, const Args&... args)
    {
        ((seed ^= std::hash<Args>{}(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2)), ...);
        return seed;
    }

    template<typename T>
    size_t vector_hash(const std::vector<T>& v)
    {
        size_t seed = v.size();
        for (const auto& val : v) {
            seed = hash_combine(seed, val);
        }
        return seed;
    }

    // Cache root shared by the join implementations.
    // File names use {func_name}_{n=...}_{params}.bin for readability and reuse.
    inline std::filesystem::path get_cache_path(const std::string& func_name,
                                                 const std::string& suffix)
    {
        static const std::filesystem::path cache_dir = "/mnt/sdc/srv/test_data_GDS";
        if (!std::filesystem::exists(cache_dir)) {
            std::filesystem::create_directories(cache_dir);
        }
        std::ostringstream oss;
        oss << func_name << "_" << suffix << ".bin";
        return cache_dir / oss.str();
    }

    // Uniqueness validation helpers.
    template<typename T>
    void validate_uniqueness(const std::vector<T>& data, const std::string& name)
    {
        if (data.size() > 1000000) {
            std::cout << name << " uniqueness check skipped for large dataset" << std::endl;
            return;
        }
        std::set<T> unique_set(data.begin(), data.end());
        if (unique_set.size() != data.size()) {
            std::cout << "WARNING: " << name << " have duplicates! "
                      << unique_set.size() << " unique out of " << data.size() << std::endl;
            std::unordered_map<T, int> count_map;
            for (const auto& item : data) count_map[item]++;
            int duplicate_count = 0;
            for (const auto& pair : count_map) {
                if (pair.second > 1) {
                    duplicate_count++;
                    if (duplicate_count <= 5) {
                        std::cout << "  Key " << pair.first << " appears " << pair.second << " times" << std::endl;
                    }
                }
            }
            if (duplicate_count > 5) {
                std::cout << "  ... and " << (duplicate_count - 5) << " more duplicates" << std::endl;
            }
        } else {
            std::cout << name << " are all unique" << std::endl;
        }
    }

    // PinnedBuffer<T>: RAII wrapper around cudaMallocHost / cudaFreeHost.
    template<typename T>
    class PinnedBuffer {
    public:
        PinnedBuffer() = default;

        explicit PinnedBuffer(size_t n) : n_(n) {
            cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&ptr_), n * sizeof(T));
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMallocHost failed: ")
                                         + cudaGetErrorString(err));
        }

        ~PinnedBuffer() { if (ptr_) cudaFreeHost(ptr_); }

        // Copying is disabled; moving is supported.
        PinnedBuffer(const PinnedBuffer&)            = delete;
        PinnedBuffer& operator=(const PinnedBuffer&) = delete;

        PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
            o.ptr_ = nullptr; o.n_ = 0;
        }
        PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = o.ptr_; n_ = o.n_; o.ptr_ = nullptr; o.n_ = 0;
            return *this;
        }

        T*     data()  const { return ptr_; }
        size_t size()  const { return n_;   }
        bool   empty() const { return n_ == 0; }

    private:
        T*     ptr_ = nullptr;
        size_t n_   = 0;
    };

    // Main generator class: CPU generation with a single sequential random engine.
    template<typename T>
    class GPUDataGenerator
    {
    private:
        int Device_idx;  // Retained for compatibility; unused by CPU generation.

        // Report progress in 10 percent increments.
        void report_progress(size_t current, size_t total, const std::string& stage) const
        {
            if (total >= 1000000) {
                static size_t last_percent = 0;
                size_t percent = current * 100 / total;
                if (percent >= last_percent + 10) {
                    std::cout << "  " << stage << " progress: " << percent << "% ("
                              << current << "/" << total << ")\n";
                    last_percent = percent;
                }
                if (current == total) last_percent = 0;
            }
        }

    public:
        GPUDataGenerator(int device_id = 0) : Device_idx(device_id) {}

        // Uniform integers.
        std::vector<T> uniform(
            size_t n,
            T min_val = std::numeric_limits<T>::min(),
            T max_val = std::numeric_limits<T>::max(),
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "uniform requires integral type");
            std::cout << "Using CPU for generating\n";

            std::ostringstream suffix;
            suffix << "n" << n << "_min" << +min_val << "_max" << +max_val << "_seed" << seed;
            auto cache_path = get_cache_path("uniform", suffix.str());

            if (std::filesystem::exists(cache_path)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<T> data;
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t size;
                    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                    data.resize(size);
                    ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
                    ifs.close();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "Loaded from cache: " << n << " elements in "
                              << duration.count() << " ms\n";
                    return data;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<T> data(n);
            std::minstd_rand rng(seed);
            std::uniform_int_distribution<T> dist(min_val, max_val);
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
                if (i % (n/10 + 1) == 0) report_progress(i, n, "Uniform");
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "CPU Uniform generation: " << n << " elements in "
                      << duration.count() << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                size_t size = data.size();
                ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ofs.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
            }
            return data;
        }

        std::vector<T> random_ints(
            size_t n,
            T min_val = std::numeric_limits<T>::min(),
            T max_val = std::numeric_limits<T>::max(),
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "random_ints requires integral type");
            std::cout << "Using CPU for generating random integers\n";

            std::ostringstream suffix;
            suffix << "n" << n << "_min" << +min_val << "_max" << +max_val << "_seed" << seed;
            auto cache_path = get_cache_path("random_ints", suffix.str());

            if (std::filesystem::exists(cache_path)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<T> data;
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t size;
                    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                    data.resize(size);
                    ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
                    ifs.close();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "Loaded from cache: " << n << " elements in "
                            << duration.count() << " ms\n";
                    return data;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<T> data(n);
            std::minstd_rand rng(seed);
            std::uniform_int_distribution<T> dist(min_val, max_val);
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
                if (i % (n/10 + 1) == 0) report_progress(i, n, "Random ints");
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "CPU Random ints generation: " << n << " elements in "
                    << duration.count() << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                size_t size = data.size();
                ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ofs.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
                ofs.close();
            }

            return data;
        }

        std::vector<T> random_ints_zipf(
            size_t n,
            double alpha,
            T min_val = std::numeric_limits<T>::min(),
            T max_val = std::numeric_limits<T>::max(),
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "random_ints_zipf requires integral type");
            if (alpha <= 0.0) throw std::invalid_argument("Zipf alpha must be > 0");
            if (min_val > max_val) throw std::invalid_argument("min_val must be <= max_val");
            if (min_val == max_val) return std::vector<T>(n, min_val);

            std::cout << "Using CPU for generating Zipf-distributed integers (alpha=" << alpha << ")\n";

            std::ostringstream suffix;
            suffix << "n" << n << "_alpha" << alpha << "_min" << +min_val << "_max" << +max_val << "_seed" << seed;
            auto cache_path = get_cache_path("random_ints_zipf", suffix.str());
            if (std::filesystem::exists(cache_path)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<T> data;
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t size;
                    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                    data.resize(size);
                    ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
                    ifs.close();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "Loaded from cache: " << n << " elements in " << duration.count() << " ms\n";
                    return data;
                }
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            uint64_t N = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val) + 1;
            const size_t MAX_BUCKETS = 1000000;
            size_t num_buckets = std::min<uint64_t>(N, MAX_BUCKETS);
            uint64_t base = N / num_buckets;
            uint64_t rem = N % num_buckets;

            std::vector<double> bucket_probs(num_buckets);
            double total = 0.0;
            for (size_t i = 0; i < num_buckets; ++i) {
                uint64_t start_rank = 1 + i * base + std::min<uint64_t>(i, rem);
                uint64_t bucket_size = base + (i < rem ? 1 : 0);
                double a = static_cast<double>(start_rank) - 0.5;
                double b = static_cast<double>(start_rank + bucket_size) - 0.5;
                double prob;
                if (std::abs(alpha - 1.0) < 1e-9) {
                    prob = std::log(b / a);
                } else {
                    prob = (std::pow(b, 1.0 - alpha) - std::pow(a, 1.0 - alpha)) / (1.0 - alpha);
                }
                bucket_probs[i] = prob;
                total += prob;
            }
            for (auto& p : bucket_probs) p /= total;

            std::mt19937_64 rng(seed);
            std::discrete_distribution<size_t> bucket_dist(bucket_probs.begin(), bucket_probs.end());
            std::uniform_int_distribution<uint64_t> inner_dist_base(0, base - 1);
            std::uniform_int_distribution<uint64_t> inner_dist_larger(0, base);

            std::vector<T> data(n);
            for (size_t i = 0; i < n; ++i) {
                size_t bucket = bucket_dist(rng);
                uint64_t start_rank = 1 + (uint64_t)bucket * base + std::min<uint64_t>(bucket, rem);
                uint64_t offset = (bucket < (size_t)rem) ? inner_dist_larger(rng)
                                                          : inner_dist_base(rng);
                uint64_t rank = start_rank + offset;
                T val = static_cast<T>(static_cast<uint64_t>(min_val) + (rank - 1));
                data[i] = val;

                if (i % (n/10 + 1) == 0) report_progress(i, n, "Zipf ints");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "CPU Zipf ints generation: " << n << " elements in " << duration.count() << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                size_t size = data.size();
                ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ofs.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
            }

            return data;
        }

        // Primary keys: unique and optionally shuffled.
        std::vector<T> primary_keys(
            size_t n,
            T start_val = 0,
            uint32_t seed = 42,
            bool shuffle = true)
        {
            static_assert(std::is_integral<T>::value, "primary_keys requires integral type");
            std::cout << "Using CPU for generating\n";

            std::ostringstream suffix;
            suffix << "n" << n << "_start" << +start_val << "_seed" << seed << "_shuffle" << shuffle;
            auto cache_path = get_cache_path("primary_keys", suffix.str());

            if (std::filesystem::exists(cache_path)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<T> data;
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t size;
                    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                    data.resize(size);
                    ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
                    ifs.close();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "Loaded from cache: " << n << " elements in "
                              << duration.count() << " ms\n";
                    return data;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<T> data(n);
            std::iota(data.begin(), data.end(), start_val);
            if (shuffle && n > 1) {
                std::minstd_rand rng(seed);
                std::shuffle(data.begin(), data.end(), rng);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "CPU Primary keys generation: " << n << " elements in "
                      << duration.count() << " ms\n";

            validate_uniqueness(data, "Primary keys");

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                size_t size = data.size();
                ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ofs.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
            }
            return data;
        }

        // Foreign keys.
        std::vector<T> foreign_keys(
            const thrust::host_vector<T>& primary_keys,
            size_t n_foreign_keys,
            double match_ratio = 1.0,
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "foreign_keys requires integral type");
            std::cout << "Using CPU for generating\n";
            if (primary_keys.empty())
                throw std::invalid_argument("Primary keys cannot be empty");

            std::vector<T> pk_vec(primary_keys.begin(), primary_keys.end());
            size_t pk_hash = vector_hash(pk_vec);
            std::ostringstream suffix;
            suffix << "n" << n_foreign_keys << "_match" << match_ratio << "_seed" << seed << "_pk" << std::hex << pk_hash;
            auto cache_path = get_cache_path("foreign_keys", suffix.str());

            if (std::filesystem::exists(cache_path)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<T> data;
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t size;
                    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                    data.resize(size);
                    ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
                    ifs.close();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "Loaded from cache: " << n_foreign_keys << " elements in "
                              << duration.count() << " ms\n";
                    return data;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<T> data(n_foreign_keys);

            std::minstd_rand rng(seed);
            std::uniform_int_distribution<size_t> idx_dist(0, pk_vec.size() - 1);
            for (size_t i = 0; i < n_foreign_keys; ++i) {
                data[i] = pk_vec[idx_dist(rng)];
                if (i % (n_foreign_keys/10 + 1) == 0) report_progress(i, n_foreign_keys, "Foreign keys");
            }

            if (match_ratio < 1.0 && match_ratio > 0.0) {
                size_t n_modify = static_cast<size_t>(n_foreign_keys * (1.0 - match_ratio));
                constexpr T flip_bit = static_cast<T>(1) << (sizeof(T) * 8 - 1);
                std::minstd_rand rng2(seed + 1);
                std::uniform_int_distribution<size_t> pos_dist(0, n_foreign_keys - 1);
                for (size_t i = 0; i < n_modify; ++i) {
                    size_t pos = pos_dist(rng2);
                    data[pos] ^= flip_bit;
                }
            }

            if (n_foreign_keys > 1) {
                std::minstd_rand rng3(seed + 2);
                std::shuffle(data.begin(), data.end(), rng3);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "CPU Foreign keys generation: " << n_foreign_keys
                      << " elements in " << duration.count() << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                size_t size = data.size();
                ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ofs.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
            }
            return data;
        }

        // Normally distributed integers.
        std::vector<T> normal_integer(
            size_t n,
            double mean = 0.0,
            double stddev = 1.0,
            T min_val = std::numeric_limits<T>::min(),
            T max_val = std::numeric_limits<T>::max(),
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "normal_integer requires integral type");
            std::cout << "Using CPU for generating\n";

            std::ostringstream suffix;
            suffix << "n" << n << "_mean" << mean << "_std" << stddev << "_min" << +min_val << "_max" << +max_val << "_seed" << seed;
            auto cache_path = get_cache_path("normal_integer", suffix.str());

            if (std::filesystem::exists(cache_path)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<T> data;
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t size;
                    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                    data.resize(size);
                    ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
                    ifs.close();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "Loaded from cache: " << n << " elements in "
                              << duration.count() << " ms\n";
                    return data;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<T> data(n);
            std::minstd_rand rng(seed);
            std::normal_distribution<double> dist(mean, stddev);
            for (size_t i = 0; i < n; ++i) {
                double val = dist(rng);
                T int_val = static_cast<T>(std::round(val));
                if (int_val < min_val) int_val = min_val;
                if (int_val > max_val) int_val = max_val;
                data[i] = int_val;
                if (i % (n/10 + 1) == 0) report_progress(i, n, "Normal integer");
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "CPU Normal integer generation: " << n << " elements in "
                      << duration.count() << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                size_t size = data.size();
                ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ofs.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
            }
            return data;
        }

        // Pinned-memory variants.

        PinnedBuffer<T> random_ints_pinned(
            size_t n,
            T min_val = std::numeric_limits<T>::min(),
            T max_val = std::numeric_limits<T>::max(),
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "random_ints_pinned requires integral type");

            std::ostringstream suffix;
            suffix << "n" << n << "_min" << +min_val << "_max" << +max_val << "_seed" << seed;
            auto cache_path = get_cache_path("random_ints", suffix.str());  // Reuse the uniform cache.

            // Cache hit: load directly into pinned host memory.
            if (std::filesystem::exists(cache_path)) {
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t cached_n;
                    ifs.read(reinterpret_cast<char*>(&cached_n), sizeof(cached_n));
                    if (cached_n == n) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        PinnedBuffer<T> buf(n);
                        ifs.read(reinterpret_cast<char*>(buf.data()), n * sizeof(T));
                        ifs.close();
                        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                        std::cout << "Loaded (pinned) from cache: " << n
                                  << " elements in " << ms << " ms\n";
                        return buf;
                    }
                }
            }

            // Cache miss: generate values and write directly into pinned host memory.
            auto t0 = std::chrono::high_resolution_clock::now();
            PinnedBuffer<T> buf(n);
            std::minstd_rand rng(seed);
            std::uniform_int_distribution<T> dist(min_val, max_val);
            for (size_t i = 0; i < n; ++i) {
                buf.data()[i] = dist(rng);
                if (i % (n/10 + 1) == 0) report_progress(i, n, "Random ints (pinned)");
            }
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "CPU Random ints (pinned) generation: " << n
                      << " elements in " << ms << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
                ofs.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(T));
            }
            return buf;
        }

        PinnedBuffer<T> random_ints_zipf_pinned(
            size_t n,
            double alpha,
            T min_val = std::numeric_limits<T>::min(),
            T max_val = std::numeric_limits<T>::max(),
            uint32_t seed = 42)
        {
            static_assert(std::is_integral<T>::value, "random_ints_zipf_pinned requires integral type");
            if (alpha <= 0.0) throw std::invalid_argument("Zipf alpha must be > 0");
            if (min_val > max_val) throw std::invalid_argument("min_val must be <= max_val");
            if (min_val == max_val) {
                PinnedBuffer<T> buf(n);
                std::fill(buf.data(), buf.data() + n, min_val);
                return buf;
            }

            std::ostringstream suffix;
            suffix << "n" << n << "_alpha" << alpha << "_min" << +min_val << "_max" << +max_val << "_seed" << seed;
            auto cache_path = get_cache_path("random_ints_zipf", suffix.str());  // Reuse the Zipf cache.

            if (std::filesystem::exists(cache_path)) {
                std::ifstream ifs(cache_path, std::ios::binary);
                if (ifs) {
                    size_t cached_n;
                    ifs.read(reinterpret_cast<char*>(&cached_n), sizeof(cached_n));
                    if (cached_n == n) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        PinnedBuffer<T> buf(n);
                        ifs.read(reinterpret_cast<char*>(buf.data()), n * sizeof(T));
                        ifs.close();
                        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                        std::cout << "Loaded (pinned) from cache: " << n
                                  << " elements in " << ms << " ms\n";
                        return buf;
                    }
                }
            }

            // Generate with the same algorithm as random_ints_zipf, writing into pinned memory.
            auto start_time = std::chrono::high_resolution_clock::now();

            uint64_t N = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val) + 1;
            const size_t MAX_BUCKETS = 1000000;
            size_t num_buckets = std::min<uint64_t>(N, MAX_BUCKETS);
            uint64_t base = N / num_buckets;
            uint64_t rem  = N % num_buckets;

            std::vector<double> bucket_probs(num_buckets);
            double total = 0.0;
            for (size_t i = 0; i < num_buckets; ++i) {
                uint64_t start_rank  = 1 + i * base + std::min<uint64_t>(i, rem);
                uint64_t bucket_size = base + (i < rem ? 1 : 0);
                double a = static_cast<double>(start_rank) - 0.5;
                double b = static_cast<double>(start_rank + bucket_size) - 0.5;
                double prob;
                if (std::abs(alpha - 1.0) < 1e-9) {
                    prob = std::log(b / a);
                } else {
                    prob = (std::pow(b, 1.0 - alpha) - std::pow(a, 1.0 - alpha)) / (1.0 - alpha);
                }
                bucket_probs[i] = prob;
                total += prob;
            }
            for (auto& p : bucket_probs) p /= total;

            std::mt19937_64 rng(seed);
            std::discrete_distribution<size_t> bucket_dist(bucket_probs.begin(), bucket_probs.end());
            std::uniform_int_distribution<uint64_t> inner_dist_base(0, base - 1);
            std::uniform_int_distribution<uint64_t> inner_dist_larger(0, base);

            PinnedBuffer<T> buf(n);
            for (size_t i = 0; i < n; ++i) {
                size_t   bucket     = bucket_dist(rng);
                uint64_t start_rank = 1 + (uint64_t)bucket * base + std::min<uint64_t>(bucket, rem);
                uint64_t offset     = (bucket < (size_t)rem) ? inner_dist_larger(rng)
                                                              : inner_dist_base(rng);
                uint64_t rank       = start_rank + offset;
                buf.data()[i]       = static_cast<T>(static_cast<uint64_t>(min_val) + (rank - 1));
                if (i % (n/10 + 1) == 0) report_progress(i, n, "Zipf ints (pinned)");
            }

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            std::cout << "CPU Zipf ints (pinned) generation: " << n
                      << " elements in " << ms << " ms\n";

            std::ofstream ofs(cache_path, std::ios::binary);
            if (ofs) {
                ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
                ofs.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(T));
            }
            return buf;
        }
    };

    // Common type aliases.
    using GPUIntGenerator = GPUDataGenerator<int>;
    using GPUUInt32Generator = GPUDataGenerator<uint32_t>;
    using GPUUInt64Generator = GPUDataGenerator<uint64_t>;

} // namespace GPURandomDataGen
