//
// Created by acminor on 9/7/21.
//

#pragma once

#include <CL/opencl.h>

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>

namespace midas::opencl::protobuf
{
    struct dim3
    {
        size_t x;
        size_t y;
        size_t z;

        dim3() : x(0), y(0), z(0)
        {
        }

        dim3(size_t _x, size_t _y, size_t _z) : x(_x), y(_y), z(_z)
        {
        }

        explicit dim3(size_t dim[3]) : x(dim[0]), y(dim[1]), z(dim[2])
        {
        }
    };

    struct dim3Ref
    {
        size_t &x;
        size_t &y;
        size_t &z;

        dim3Ref(size_t &_x, size_t &_y, size_t &_z) : x(_x), y(_y), z(_z)
        {
        }

        explicit dim3Ref(size_t dim[3]) : x(dim[0]), y(dim[1]), z(dim[2])
        {
        }

        void setFromDim3(dim3 d3)
        {
            this->x = d3.x;
            this->y = d3.y;
            this->z = d3.z;
        }
    };

    class Dim3Converter : public IConverter<Dim3Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const dim3 &in, protobuf_support::pb_dim3 &out, ConvertOptions convertOptions = {})
        {
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const dim3 in, protobuf_support::pb_dim3 *out, ConvertOptions convertOptions = {})
        {
            dim3 result;
            result = in;

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const cl_mem_wrapper<size_t> in, protobuf_support::pb_dim3 *out,
                           ConvertOptions convertOptions = {})
        {
            size_t data[3];
            CLReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(data), data);

            out->set_x(data[0]);
            out->set_y(data[1]);
            out->set_z(data[2]);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void DeserializeBase(dim3Ref out, const protobuf_support::pb_dim3 &in, ConvertOptions convertOptions = {})
        {
            out.setFromDim3({(unsigned int)in.x(), (unsigned int)in.y(), (unsigned int)in.z()});
        }

        template <typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<size_t> out, const protobuf_support::pb_dim3 &in,
                             ConvertOptions convertOptions)
        {
            size_t data[3];
            this->template DeserializeBase(dim3Ref(data), in, convertOptions);
            CLWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(data), data);
        }
    };

    RegisterCudaConverter1(Dim3Converter);
} // namespace midas::opencl::protobuf
