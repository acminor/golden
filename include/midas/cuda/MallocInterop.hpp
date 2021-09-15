//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MALLOCINTEROP_HPP
#define GOLDEN_MALLOCINTEROP_HPP

#include <cuda.h>
#include <unordered_map>

#define MIDAS_CUDA_MALLOC_INTEROP_ENABLED

std::unordered_map<void *, size_t> cudaMemoryMappings;

__host__ cudaError_t internalCudaMallocRedfine(void **devPtr, size_t size)
{
    auto error = cudaMalloc(devPtr, size);

    if (error == cudaSuccess)
        cudaMemoryMappings[*devPtr] = size;

    return error;
}

__host__ cudaError_t internalCudaFreeRedefine(void *devPtr)
{
    auto error = cudaFree(devPtr);

    if (error == cudaSuccess)
        cudaMemoryMappings.erase(devPtr);
}

size_t cudaGetArraySize(void *devPtr)
{
    auto result = cudaMemoryMappings.find(devPtr);

    // std::cout << result->second << std::endl;

    return result != std::end(cudaMemoryMappings) ? result->second : 0;
}

#define cudaMalloc(DEV_PTR, SIZE) internalCudaMallocRedfine(DEV_PTR, SIZE)
#define cudaFree(DEV_PTR) internalCudaFreeRedefine(DEV_PTR)

#endif // GOLDEN_MALLOCINTEROP_HPP
