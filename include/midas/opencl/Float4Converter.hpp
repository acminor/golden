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
    class Float4Converter : public IConverter<Float4Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_float4> mem, protobuf_support::pb_float4 *out,
                           ConvertOptions convertOptions)
        {
            cl_float4 data;
            CLReadBuffer<ConvertOptions::MemoryOption>(mem, data);
            this->SerializeBase(data, out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_float4> mem, protobuf_support::pb_float4 &out,
                           ConvertOptions convertOptions)
        {
            this->template SerializeBase(mem, &out, convertOptions);
        }

        template <typename ConvertOptions>
        void SerializeBase(const cl_float4 &in, protobuf_support::pb_float4 &out, ConvertOptions convertOptions)
        {
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions>
        void SerializeBase(const cl_float4 &in, protobuf_support::pb_float4 *out, ConvertOptions convertOptions)
        {
            out->set_x(in.x);
            out->set_y(in.y);
            out->set_z(in.z);
            out->set_w(in.w);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<cl_float4> out, const protobuf_support::pb_float4 &in,
                             ConvertOptions convertOptions)
        {
            cl_float4 host;
            this->DeserializeBase(host, in, convertOptions);

            CLWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(cl_float4), &host);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_float4 &out, const protobuf_support::pb_float4 &in, ConvertOptions convertOptions)
        {
            out = {in.x(), in.y(), in.z(), in.w()};
        }
    };

    RegisterCudaConverter1(Float4Converter);
} // namespace midas::opencl::protobuf