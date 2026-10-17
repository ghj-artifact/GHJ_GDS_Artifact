# GHJ_GDS Artifact

This repository contains the source for the two current join entry points:

- `GHJ_GDS_WPS`: warp-shuffle partitioning with the counting-hash join pipeline.
- `GHJ_GDS_SMH`: shared-memory atomic-histogram partitioning with the same join pipeline.

Both programs support uniform random data, Zipf-distributed data, and TPC-H
join-key workloads. GDS/cuFile is used for partitioned data and spill files.

## Requirements

- Linux
- CMake 3.24 or newer
- A C++17-capable host compiler
- NVIDIA CUDA Toolkit with `nvcc`
- cuFile development headers and library
- An NVIDIA GPU and GDS-capable storage for performance runs

The build uses `CMAKE_CUDA_ARCHITECTURES=native` by default. Set an explicit
architecture when cross-compiling or when the build host has no visible GPU. For
example, use `90` for H100 and `89` for Ada GPUs.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

To select the target GPU architecture explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --parallel
```

The executables are written to `build/bin/`:

```text
build/bin/GHJ_GDS_WPS
build/bin/GHJ_GDS_SMH
build/bin/cufile_gds4m.json
```

The CMake targets are:

| Target | Purpose |
| --- | --- |
| `GHJ_GDS_WPS` | Warp-shuffle partitioner and counting-hash join pipeline |
| `GHJ_GDS_SMH` | Atomic-histogram partitioner and counting-hash join pipeline |
| `ghj_data_generator` | Uniform and Zipf input generation |
| `ghj_partitioner_wps` | Warp-shuffle partitioner |
| `ghj_partitioner_smh` | Shared-memory atomic-histogram partitioner |
| `ghj_joiner` | Counting-hash join and read pipeline |

## Source layout

| Path | Purpose |
| --- | --- |
| `GHJ_GDS/GHJ_GDS_WPS.cu` | Main program with the warp-shuffle partitioner |
| `GHJ_GDS/GHJ_GDS_SMH.cu` | Main program with the histogram partitioner |
| `GHJ_GDS/Partitioner/gpu_partitioner_wps.cu` | WPS partitioning and spill pipeline |
| `GHJ_GDS/Partitioner/gpu_partitioner_smh.cu` | SMH partitioning and spill pipeline |
| `GHJ_GDS/Joiner/gpu_hash_join.cu` | Counting-hash build/probe kernels and chunked reads |
| `GHJ_GDS/cufile_gds4m.json` | cuFile configuration with 4 MiB internal I/O chunks |
| `Data_Generate/` | Random/Zipf generators and TPC-H key-cache loader |

## Runtime setup

Run the executable from `build/bin` so that it finds the adjacent cuFile
configuration, or set the configuration path explicitly:

```bash
export CUFILE_ENV_PATH_JSON="$PWD/build/bin/cufile_gds4m.json"
```

Select the CUDA device:

```bash
export CUDA_VISIBLE_DEVICES=0
export GHJ_DEVICEID=0
```

If the host uses MPS, point the process at an isolated MPS directory:

```bash
export CUDA_MPS_PIPE_DIRECTORY="$(mktemp -d)"
```

The default spill paths in the source are host-specific. Set `GHJ_LAND` to
writable GDS-capable directories before running. The format is a comma-separated
list of `directory[:slots[:lanes]]` entries:

```bash
export GHJ_LAND="/path/to/gds-a:4:4,/path/to/gds-b:4:4"
```

Repeated directories act as weights. Each run expects exclusive use of its spill
directories because partition files are overwritten.

## Run

Uniform data:

```bash
cd build/bin
./GHJ_GDS_WPS 16 1000000 1000000 0.0 12 4
```

Zipf data:

```bash
./GHJ_GDS_WPS 16 1000000 1000000 0.8 12 4
```

Histogram partitioner:

```bash
./GHJ_GDS_SMH 16 1000000 1000000 0.8 12 4
```

TPC-H key cache:

```bash
./GHJ_GDS_WPS --tpch Q3 100 all 32 12 4
./GHJ_GDS_SMH --tpch Q5 100 all 32 12 4
```

The random/Zipf argument order is:

```text
partition_count data_size_R data_size_S zipf_alpha target_batches num_workers
```

The TPC-H argument order is:

```text
--tpch <Q2|Q3|Q4|Q5> <SF> <join_name|all> <partition_count> <target_batches> <num_workers>
```

## Runtime options

| Variable | Meaning |
| --- | --- |
| `GHJ_DEVICEID` | CUDA ordinal used by the program |
| `GHJ_LAND` | Spill layout in `directory:slots:lanes` form |
| `GHJ_CHUNK_MB` | Read-chunk size in MiB; `0` reads whole partitions |
| `GHJ_CHUNK_READERS` | Number of chunks read concurrently per round |
| `GHJ_PREFETCH` | `1` enables cross-partition prefetch |
| `GHJ_RS_PARALLEL` | `1` reads R and S concurrently for a partition |
| `GHJ_CUFILE_4M` | `0` disables the bundled cuFile tuning |
| `GHJ_CUFILE_JSON` | Alternative path to a cuFile JSON configuration |
| `CUFILE_ENV_PATH_JSON` | cuFile configuration file selected for the process |
| `CUDA_MPS_PIPE_DIRECTORY` | Isolated directory for an MPS client |

The reported performance metric uses the program's preprocess, join, and total
timings. Input loading is outside that metric.

## Data format

TPC-H cache files use the following layout:

```text
[8-byte little-endian element count][count x uint32_t keys]
```

File names follow:

```text
tpch_<query>_SF<scale_factor>_<join_name>_<R|S>.bin
```

The data-generation headers define the random and Zipf cache naming scheme and
the expected cache root. Prepare compatible cache files before running a
workload.

## Notes

- `GHJ_GDS_WPS` and `GHJ_GDS_SMH` share the join implementation but use
  different partitioning kernels.
- Four MiB cuFile chunks are the expected configuration for reported
  experiments.
- Cross-run comparisons should use the same CUDA device, input files, spill
  layout, cuFile configuration, and read options.

## License

Original material in this repository is provided under the MIT License. See
`LICENSE`. External dependencies, the bundled cuFile configuration, and TPC-H
material remain subject to their own terms; see `THIRD_PARTY_NOTICES.md`.
