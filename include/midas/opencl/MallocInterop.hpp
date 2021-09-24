//
// Created by acminor on 9/7/21.
//

#pragma once

#include <CL/opencl.h>
#include <unordered_map>

#define MIDAS_OPENCL_MALLOC_INTEROP_ENABLED

std::unordered_map<cl_mem, size_t> openclMemoryMappings;

cl_mem internalClCreateBufferRedefine(cl_context context, cl_mem_flags flags, size_t size, void *hostData, cl_int *err)
{
    auto mem = clCreateBuffer(context, flags, size, hostData, err);

    if (*err == CL_SUCCESS)
        openclMemoryMappings[mem] = size;

    return mem;
}

size_t openclGetArraySize(cl_mem mem)
{
    auto result = openclMemoryMappings.find(mem);
    return result != std::end(openclMemoryMappings) ? result->second : 0;
}

#define clCreateBuffer(...) internalClCreateBufferRedefine(__VA_ARGS__)