//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_CUDA_UTILITY_HPP
#define GOLDEN_CUDA_UTILITY_HPP

#include <type_traits>

#include <cuda.h>

namespace midas::cuda::utility
{
    struct CudaMemoryType
    {
    };

    struct CudaNormalMemory : CudaMemoryType
    {
    };

    struct CudaSymbolMemory : CudaMemoryType
    {
    };

    template <typename T, typename CudaMemoryType = CudaNormalMemory>
    void hostToCuda(T **cudaMemory, const std::vector<T> &hostMemory)
    {
        cudaMalloc((void **)cudaMemory, hostMemory.size() * sizeof(T));

        if constexpr (std::is_same_v<CudaMemoryType, CudaNormalMemory>)
            cudaMemcpy((void **)cudaMemory, hostMemory.data(), hostMemory.size() * sizeof(T), cudaMemcpyHostToDevice);
        else if constexpr (std::is_same_v<CudaMemoryType, CudaSymbolMemory>)
            cudaMemcpyToSymbol((void **)cudaMemory, hostMemory.data(), hostMemory.size() * sizeof(T),
                               cudaMemcpyHostToDevice);
    }

    template <typename T, typename CudaMemoryType = CudaNormalMemory>
    void hostToCuda(T **cudaMemory, const T &hostMemory)
    {
        cudaMalloc((void **)cudaMemory, sizeof(T));

        if constexpr (std::is_same_v<CudaMemoryType, CudaNormalMemory>)
            cudaMemcpy((void **)cudaMemory, &hostMemory, sizeof(T), cudaMemcpyHostToDevice);
        else if constexpr (std::is_same_v<CudaMemoryType, CudaSymbolMemory>)
            cudaMemcpyToSymbol((void **)cudaMemory, &hostMemory, sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T, typename CudaMemoryType = CudaNormalMemory>
    std::vector<T> &&cudaToHost(const T *mem, size_t numberOfElements)
    {
        std::vector<T> x(numberOfElements);

        if constexpr (std::is_same_v<CudaMemoryType, CudaNormalMemory>)
            cudaMemcpy(x.data(), mem, numberOfElements * sizeof(T), cudaMemcpyDeviceToHost);
        else if constexpr (std::is_same_v<CudaMemoryType, midas::cuda::utility::CudaSymbolMemory>)
            cudaMemcpyFromSymbol(x.data(), mem, numberOfElements * sizeof(T), cudaMemcpyDeviceToHost);

        return std::move(x);
    }

    template <typename T, typename CudaMemoryType = CudaNormalMemory> T &&cudaToHost(const T &mem)
    {
        T data;

        if constexpr (std::is_same_v<CudaMemoryType, CudaNormalMemory>)
            cudaMemcpy(data, mem, sizeof(T), cudaMemcpyDeviceToHost);
        else if constexpr (std::is_same_v<CudaMemoryType, midas::cuda::utility::CudaSymbolMemory>)
            cudaMemcpyFromSymbol(data, mem, sizeof(T), cudaMemcpyDeviceToHost);

        return std::move(data);
    }
} // namespace midas::cuda::utility

#endif // GOLDEN_CUDA_UTILITY_HPP
