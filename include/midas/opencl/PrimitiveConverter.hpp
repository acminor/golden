//
// Created by acminor on 9/8/21.
//

#pragma once

#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>

namespace midas::opencl::protobuf
{
    class PrimitiveConverter : public IConverter<PrimitiveConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const HostType &in, SerialOutFunction out, ConvertOptions convertOptions)
        {
            out(in);
        }

        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const cl_mem_wrapper<HostType> in, SerialOutFunction out, ConvertOptions convertOptions)
        {
            HostType result;
            CLReadBuffer<ConvertOptions::MemoryOption>(in, result);
            out(result);
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<HostType> out, const SerialType &in, ConvertOptions convertOptions)
        {
            HostType data;
            this->DeserializeBase(data, in, convertOptions);
            CLWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(HostType), &data);
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(HostType &out, const SerialType &in, ConvertOptions convertOptions)
        {
            out = in;
        }
    };

    RegisterCudaConverter1(PrimitiveConverter);
} // namespace midas::opencl::protobuf