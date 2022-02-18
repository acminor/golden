//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/Converters.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>
#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/VectorConverter.hpp>

size_t openclGetArraySize(cl_mem);
namespace midas::opencl::protobuf
{
    class MallocInteropArrayConverter : public IConverter<MallocInteropArrayConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(cl_mem_wrapper<HostType> in, SerialOutFunction out, ConvertOptions options)
        {
            auto size = openclGetArraySize(in.m_mem) / sizeof(HostType);

            if (size != 0)
                converters::VectorConverter.Serialize(std::make_pair(in, size), out, options);
            else
                throw ConverterException("Ahhh, using memory buffer without size and not tracked");
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(HostType, SerialType, ConvertOptions)
        {
            throw ConverterException("Not implemented");
        }
    };

    RegisterCudaConverter1(MallocInteropArrayConverter);
} // namespace midas::opencl::protobuf