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
    class Int4Converter : public IConverter<Int4Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_int4> mem, protobuf_support::pb_int4 *out,
                           ConvertOptions convertOptions)
        {
            cl_int4 data;
            CLReadBuffer<ConvertOptions::MemoryOption>(mem, data);
            this->SerializeBase(data, out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<cl_int4> mem, protobuf_support::pb_int4 &out,
                           ConvertOptions convertOptions)
        {
            this->template SerializeBase(mem, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const cl_int4 &in, protobuf_support::pb_int4 &out, ConvertOptions convertOptions = {})
        {
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const cl_int4 &in, protobuf_support::pb_int4 *out, ConvertOptions convertOptions = {})
        {
            out->set_x(in.x);
            out->set_y(in.y);
            out->set_z(in.z);
            out->set_w(in.w);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<cl_int4> out, const protobuf_support::pb_int4 &in,
                             ConvertOptions convertOptions)
        {
            cl_int4 host;
            this->DeserializeBase(host, in, convertOptions);

            CLWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(cl_int4), &host);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void DeserializeBase(cl_int4 &out, const protobuf_support::pb_int4 &in, ConvertOptions convertOptions = {})
        {
            out = {(int)in.x(), (int)in.y(), (int)in.z(), (int)in.w()};
        }
    };

    RegisterCudaConverter1(Int4Converter);
} // namespace midas::opencl::protobuf