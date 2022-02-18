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
midas::opencl::protobuf
{
    class Float3Converter : public IConverter<Float3Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_float3> mem, protobuf_support::pb_float3 *out,
                           ConvertOptions convertOptions)
        {
            cl_float3 data;
            CLReadBuffer<ConvertOptions::MemoryOption>(mem, data);
            this->SerializeBase(data, out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_float3> mem, protobuf_support::pb_float3 &out,
                           ConvertOptions convertOptions)
        {
            this->SerializeBase(mem, &out, convertOptions);
        }

        template <typename ConvertOptions>
        void SerializeBase(const cl_float3 &in, protobuf_support::pb_float3 &out, ConvertOptions convertOptions)
        {
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions>
        void SerializeBase(const cl_float3 &in, protobuf_support::pb_float3 *out, ConvertOptions convertOptions)
        {
            out->set_x(in.x);
            out->set_y(in.y);
            out->set_z(in.z);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<cl_float3> out, const protobuf_support::pb_float3 &in,
                             ConvertOptions convertOptions)
        {
            cl_float3 host;
            this->DeserializeBase(host, in, convertOptions);

            CLWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(cl_float3), &host);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_float3 &out, const protobuf_support::pb_float3 &in, ConvertOptions convertOptions)
        {
            out = {in.x(), in.y(), in.z()};
        }
    };

    RegisterCudaConverter1(Float3Converter);
} // namespace midas::opencl::protobuf