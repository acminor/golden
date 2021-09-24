//
// Created by acminor on 9/20/21.
//

#pragma once

#include <CL/opencl.h>

#include <optional>

namespace midas::opencl
{
    std::optional<cl_command_queue> MidasDefaultCommandQueue;
    std::optional<cl_device_id> MidasDefaultDeviceId;
    std::optional<cl_platform_id> MidasDefaultPlatformId;
    std::optional<cl_context> MidasDefaultContext;

    void InitializeDefaults(cl_command_queue defaultQueue, cl_device_id defaultDeviceId,
                            cl_platform_id defaultPlatformId, cl_context defaultContext)
    {
        MidasDefaultCommandQueue = std::make_optional(defaultQueue);
        MidasDefaultDeviceId = std::make_optional(defaultDeviceId);
        MidasDefaultPlatformId = std::make_optional(defaultPlatformId);
        MidasDefaultContext = std::make_optional(defaultContext);
    }

    template <typename T = void>
    class cl_mem_wrapper
    {
      public:
        static constexpr char Tag[] = "cl_mem_wrapper";

        using UnderlyingType = T;

        using OptionalContext = std::optional<cl_context>;
        using OptionalQueue = std::optional<cl_command_queue>;

        cl_mem_wrapper(cl_mem &mem, cl_mem_flags flags = CL_MEM_READ_WRITE, OptionalContext context = {},
                       OptionalQueue queue = {})
            : m_mem(mem), m_context(context), m_flags(flags), m_commandQueue(queue)
        {
        }

        cl_mem &m_mem;
        cl_mem_flags m_flags;
        OptionalContext m_context;
        OptionalQueue m_commandQueue;
    };
} // namespace midas::opencl