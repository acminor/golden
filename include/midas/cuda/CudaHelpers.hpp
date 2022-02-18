//
// Created by acminor on 9/27/21.
//

#pragma once

#include <chrono>
#include <cstring>
#include <cuda.h>
#include <thread>

namespace midas::cuda
{
    template <MemoryOptions MemoryOption>
    struct StateInMemoryOptions
    {
        static constexpr bool value = MemoryOption == MemoryOptions::Host || MemoryOption == MemoryOptions::Device ||
                                      MemoryOption == MemoryOptions::Symbol;
    };

    template <MemoryOptions MemoryOption, typename T>
    void CudaReadBuffer(const T *mem, size_t size, T *data)
    {
        static_assert(StateInMemoryOptions<MemoryOption>().value);
        if constexpr (MemoryOption == MemoryOptions::Host || MemoryOption == MemoryOptions::HostAlloc)
        {
            memcpy(data, mem, size);
        }
        else if constexpr (MemoryOption == MemoryOptions::Device || MemoryOption == MemoryOptions::DeviceNoAlloc)
        {
            cudaMemcpy(data, mem, size, cudaMemcpyDeviceToHost);
        }
        else if constexpr (MemoryOption == MemoryOptions::Symbol)
        {
            cudaMemcpyFromSymbol(data, *mem, size);
        }
    }

    template <MemoryOptions MemoryOption, typename T>
    void CudaAllocateBuffer(T **mem, size_t size)
    {
        static_assert(MemoryOption != MemoryOptions::Symbol, "Symbol allocation is not currently supported.");
        if constexpr (MemoryOption == MemoryOptions::Host || MemoryOption == MemoryOptions::HostAlloc)
        {
            *mem = (T *)malloc(size);
        }
        else if constexpr (MemoryOption == MemoryOptions::Device || MemoryOption == MemoryOptions::DeviceNoAlloc)
        {
            cudaMalloc((void **)mem, size);
        }
    }

    template <MemoryOptions MemoryOption, typename T>
    void CudaMemcpyBuffer(T *mem, size_t size, const T *data)
    {
        if constexpr (MemoryOption == MemoryOptions::Host || MemoryOption == MemoryOptions::HostAlloc)
        {
            memcpy(mem, data, size);
        }
        else if constexpr (MemoryOption == MemoryOptions::Device || MemoryOption == MemoryOptions::DeviceNoAlloc)
        {
            cudaMemcpy(mem, data, size, cudaMemcpyHostToDevice);
        }
        else if constexpr (MemoryOption == MemoryOptions::Symbol)
        {
            // TODO need logic to handle C vs CXX api for cudaMemcpyToSymbol
            // - these have 2 different apis and will still compile but work incorrectly
            cudaMemcpyToSymbol(*mem, data, size);
        }
    }

    template <MemoryOptions MemoryOption, typename T>
    void CudaWriteBuffer(T *mem, size_t size, const T *data)
    {
        static_assert(!IsAllocOption(MemoryOption));
        CudaMemcpyBuffer<MemoryOption>(mem, size, data);
    }

    template <MemoryOptions MemoryOption, typename T>
    void CudaWriteBuffer(T **mem, size_t size, const T *data)
    {
        static_assert(IsAllocOption(MemoryOption));
        CudaAllocateBuffer<MemoryOption>(mem, size);
        CudaMemcpyBuffer<MemoryOption>(*mem, size, data);
    }

    template <MemoryOptions MemoryOption, typename T>
    void CudaWriteBuffer(T **mem, size_t size, const T **data)
    {
        static_assert(!IsAllocOption(MemoryOption));
        CudaMemcpyBuffer<MemoryOption>(mem, size, data);
    }
} // namespace midas::cuda
