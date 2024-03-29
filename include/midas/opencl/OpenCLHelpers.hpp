//
// Created by acminor on 9/21/21.
//

#pragma once

#include <CL/opencl.h>
#include <midas/opencl/OpenCLBase.hpp>

namespace midas::opencl
{
#ifdef MIDAS_OPENCL_MALLOC_INTEROP_ENABLED
    void OpenCLEraseAllMemory()
    {
        while (!openclMemoryMappings.empty())
        {
            auto it = openclMemoryMappings.begin();
            auto err = clReleaseMemObject(it->first);
            openclMemoryMappings.erase(it);
        }
    }
#endif

    template <MemoryOptions MemoryOption, typename T>
    void CLReadBuffer(const cl_mem_wrapper<T> &mem, size_t size, T *data)
    {
        static_assert(MemoryOption != MemoryOptions::Host, "Cannot use host with cl_mem_wrapper.");
        static_assert(MemoryOption != MemoryOptions::Host || MemoryOption != MemoryOptions::HostAlloc,
                      "Cannot use host with cl_mem_wrapper.");
        static_assert(MemoryOption != MemoryOptions::Symbol, "Symbol memory on OpenCL is not yet supported.");

        auto commandQueue = GetProperCommandQueue(mem);
        clEnqueueReadBuffer(commandQueue, mem.m_mem, true, 0, size, (void *)data, 0, nullptr, nullptr);
        clFlush(commandQueue);
        clFinish(commandQueue);
    }

    template <MemoryOptions MemoryOption, typename T>
    void CLReadBuffer(const cl_mem_wrapper<T> &mem, T &data)
    {
        CLReadBuffer<MemoryOption>(mem, sizeof(T), &data);
    }

    template <MemoryOptions MemoryOption, typename T>
    void CLWriteBuffer(cl_mem_wrapper<T> &mem, size_t size, const T *data)
    {
        static_assert(MemoryOption != MemoryOptions::Host || MemoryOption != MemoryOptions::HostAlloc,
                      "Cannot use host with cl_mem_wrapper.");
        static_assert(MemoryOption != MemoryOptions::Symbol, "Symbol memory on OpenCL is not yet supported.");

        auto context = GetProperContext(mem);
        auto commandQueue = GetProperCommandQueue(mem);
        cl_int err;

        if constexpr (MemoryOption == MemoryOptions::Device)
            mem.m_mem = clCreateBuffer(context, mem.m_flags, size, nullptr, &err);

        clEnqueueWriteBuffer(commandQueue, mem.m_mem, true, 0, size, (void *)data, 0, nullptr, nullptr);
        clFlush(commandQueue);
        clFinish(commandQueue);
    }

    template <typename T>
    cl_context GetProperContext(const cl_mem_wrapper<T> &mem)
    {
        if (mem.m_context)
        {
            return *mem.m_context;
        }
        else if (MidasDefaultContext)
        {
            return *MidasDefaultContext;
        }
        else
        {
            throw "TODO exception";
        }
    }

    template <typename T>
    cl_command_queue GetProperCommandQueue(const cl_mem_wrapper<T> &mem)
    {
        if (mem.m_commandQueue)
        {
            return *mem.m_commandQueue;
        }
        else if (MidasDefaultCommandQueue)
        {
            return *MidasDefaultCommandQueue;
        }
        else
        {
            throw "TODO exception";
        }
    }
} // namespace midas::opencl
