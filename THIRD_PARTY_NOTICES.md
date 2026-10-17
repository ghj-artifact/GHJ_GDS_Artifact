# Third-Party Notices

This repository contains original project source together with configuration
material that depends on external software. The MIT license in `LICENSE`
applies only to original material in this repository. Third-party components
remain subject to their own licenses and terms.

No third-party source tree is vendored in this repository.

## NVIDIA CUDA Toolkit and cuFile

The program is compiled against the NVIDIA CUDA Runtime and the cuFile library.
CUDA and cuFile are not distributed in this repository. Obtain them from NVIDIA
and comply with the applicable NVIDIA license terms.

`GHJ_GDS/cufile_gds4m.json` follows the configuration format and default
settings supplied with NVIDIA cuFile. Its project-specific change sets
`max_direct_io_size_kb` to 4096 so that internal cuFile requests use 4 MiB
chunks. The file is provided as a runtime configuration aid and does not grant
rights to NVIDIA software. NVIDIA, CUDA, and cuFile are trademarks or registered
trademarks of NVIDIA Corporation.

## NVIDIA Thrust

`Data_Generate/gpu_data_generator.cuh` uses the Thrust header-only library from
the CUDA Toolkit. Thrust source code is not distributed in this repository.
Obtain it through the CUDA Toolkit and comply with the terms of that
distribution.

## TPC-H

The programs can read join-key caches derived from TPC-H benchmark data. TPC-H
data, specifications, and trademarks are not distributed in this repository.
Users are responsible for obtaining TPC-H material from the Transaction
Processing Performance Council and following its usage terms.

## System Libraries

The build may link against the platform C++ runtime, POSIX threading support,
and other libraries supplied by the operating system. Those libraries remain
under their respective licenses.
