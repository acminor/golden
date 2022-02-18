//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/ProtobufUtility.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>
#include <midas/opencl/OpenCLBase.hpp>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>

namespace midas::opencl::protobuf
{
    using namespace midas::protobuf;

    class VectorConverter : public IConverter<VectorConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const std::string &in, SerialOutFunction out, ConvertOptions options)
        {
            out(in);
        }

        template <typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const std::pair<cl_mem_wrapper<char>, size_t> in, SerialOutFunction out,
                           ConvertOptions options)
        {
            auto mem = in.first;
            auto data = in.first.m_mem;
            auto length = in.second;
            auto byteSize = sizeof(char) * length;

            auto buffer = std::make_unique::<char *>(length);
            CLReadBuffer<ConvertOptions::MemoryOption>(mem, byteSize, buffer.get());
            auto dataString = std::string(buffer.get(), length);

            out(dataString);
        }

        template <typename HostType, typename ConvertOptions>
        void DeserializeBase(std::string &hostMem, std::string snapshotField, ConvertOptions options)
        {
            hostMem(snapshotField);
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<char> mem, std::string snapshotField, ConvertOptions options)
        {
            const auto byteSize = hostVector.size() * sizeof(char);
            CLWriteBuffer<ConvertOptions::MemoryOption>(mem, byteSize, hostVector.data());
        }
    };

    RegisterCudaConverter1(VectorConverter);
} // namespace midas::opencl::protobuf