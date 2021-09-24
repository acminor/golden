//
// Created by acminor on 9/23/21.
//

#include "TestsUtility.hpp"

//
// Created by acminor on 9/23/21.
//

#pragma once

#include <iostream>
#include <vector>

#define EXIT_FAIL(MSG)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cout << MSG << std::endl;                                                                                 \
        exit(-1);                                                                                                      \
    } while (0)

struct OpenClData OpenClData;

void InitializeOpenCL()
{
    static bool isInitialized = false;
    if (isInitialized)
        return;

    cl_uint num_of_platforms;
    clGetPlatformIDs(0, nullptr, &num_of_platforms);
    std::vector<cl_platform_id> platformIds(num_of_platforms);
    if (clGetPlatformIDs(num_of_platforms, platformIds.data(), nullptr) != CL_SUCCESS)
        EXIT_FAIL("Unable to get any platform_ids");
    cl_platform_id platform = platformIds[0];
    cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_int err;
    cl_device_id deviceId;

    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &deviceId, nullptr) != CL_SUCCESS)
        EXIT_FAIL("Unable to get device_id");

    cl_context context = clCreateContext(properties, 1, &deviceId, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
        EXIT_FAIL("Unable to create context");

    cl_command_queue_properties queueProperties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceId, queueProperties, &err);
    if (err != CL_SUCCESS)
        EXIT_FAIL("Unable to create command queue");

    OpenClData.queue = queue;
    OpenClData.context = context;
    OpenClData.platform = platform;
    OpenClData.device = deviceId;

    isInitialized = true;
}

cl_mem easyBufferCreate(void *data, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE)
{
    InitializeOpenCL();

    cl_int err;
    cl_mem mem = clCreateBuffer(OpenClData.context, flags, size, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Unable to create buffer" << std::endl;
        exit(-1);
    }
    err = clEnqueueWriteBuffer(OpenClData.queue, mem, true, 0, size, (void *)&data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        EXIT_FAIL("Unable to memcpy buffer");

    err = clFlush(OpenClData.queue);
    if (err != CL_SUCCESS)
        EXIT_FAIL("Unable to flush memcpy buffer");

    err = clFinish(OpenClData.queue);
    if (err != CL_SUCCESS)
        EXIT_FAIL("Unable to finish memcpy buffer");

    return mem;
}
