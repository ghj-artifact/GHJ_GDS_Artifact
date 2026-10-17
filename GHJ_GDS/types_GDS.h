/*
types_GDS.h - shared types and hashing helpers

  intkey_t = uint32_t (key type), value_t = uint32_t
  Entry / Slot / Joiner - chained hash-table structures shared by legacy paths
  fnv1a_hash() / hash_func() - partitioning and hashing helpers

Included by the partitioner, joiner, and main programs.
*/
#ifndef TYPES_GDS_H
#define TYPES_GDS_H

#include <stdint.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>
using namespace std;

#define BUCKET_SIZE 100
#define NUM_BUCKET 1000000
#define NUM_TUPLES 100000
#define Empty 0xffffffff
#define HASH_SCALE_FACTOR 2

typedef uint32_t intkey_t;
typedef uint32_t value_t;

struct Entry {
    intkey_t key;
    value_t value;
    Entry* next;
} __attribute__((aligned(16))); // Align entries to improve memory access.

// Hash bucket containing the head of a chain.
struct Slot {
    Entry* head;
};

// Hash-join state with pointers to the bucket and entry arrays.
struct Joiner {
    Slot* slots;
    Entry* entries;
    int table_size;
    int num_entries;
};

// 32-bit FNV-1a hash function.
__device__ __host__ inline uint32_t fnv1a_hash(intkey_t key) {
    const uint32_t FNV_OFFSET_BASIS = 2166136261U;
    const uint32_t FNV_PRIME = 16777619U;
    uint32_t hash = FNV_OFFSET_BASIS;
    uint8_t* bytes = (uint8_t*)&key;
    for (int i = 0; i < sizeof(intkey_t); ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

__device__ __host__ inline int hash_func(intkey_t key, int table_size) {
    return fnv1a_hash(key) % table_size;
}

#endif
