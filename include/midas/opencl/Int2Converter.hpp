//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>
#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>

#include <CL/opencl.h>

namespace midas::opencl::protobuf
{
    class Int2Converter : public IConverter<Int2Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_int2> mem, protobuf_support::pb_int2 *out,
                           ConvertOptions convertOptions)
        {
            cl_int2 data;
            CLReadBuffer<ConvertOptions::MemoryOption>(mem, data);
            this->SerializeBase(data, out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_int2> mem, protobuf_support::pb_int2 &out,
                           ConvertOptions convertOptions)
        {
            this->SerializeBase(mem, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_int2 &in, protobuf_support::pb_int2 &out, ConvertOptions convertOptions = {})
        {
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_int2 &in, protobuf_support::pb_int2 *out, ConvertOptions convertOptions = {})
        {
            out->set_x(in.x);
            out->set_y(in.y);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<cl_int2> out, const protobuf_support::pb_int2 &in,
                             ConvertOptions convertOptions)
        {
            cl_int2 host;
            this->DeserializeBase(host, in, convertOptions);

            CLWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(cl_int2), &host);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void DeserializeBase(cl_int2 &out, const protobuf_support::pb_int2 &in, ConvertOptions convertOptions = {})
        {
            out = {(int)in.x(), (int)in.y()};
        }
    };

    RegisterCudaConverter1(Int2Converter);
} // namespace midas::opencl::protobuf