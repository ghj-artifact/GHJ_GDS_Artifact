/*
gpu_data_generator.cu - random data generation (GPU template instantiations)
Compiled as one of the main program translation units.
*/
#include "gpu_data_generator.cuh"

// Explicit instantiations make the template implementations available to the linker.
template class GPURandomDataGen::GPUDataGenerator<int>;
template class GPURandomDataGen::GPUDataGenerator<uint32_t>;
template class GPURandomDataGen::GPUDataGenerator<uint64_t>;
