//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/ProtobufUtility.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>
#include <midas/opencl/OpenCLBase.hpp>

#include <optional>
#include <type_traits>

namespace midas::opencl::protobuf
{
    using namespace midas::protobuf;

    class VectorConverter : public IConverter<VectorConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const std::vector<HostType> &in, SerialOutFunction out, ConvertOptions options)
        {
            for (auto x : in)
                out(x);
        }

        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const std::pair<cl_mem_wrapper<HostType>, size_t> in, SerialOutFunction out,
                           ConvertOptions options)
        {
            auto mem = in.first;
            auto data = in.first.m_mem;
            auto length = in.second;
            auto byteSize = sizeof(HostType) * length;

            std::vector<HostType> dataVector(length);
            CLReadBuffer<ConvertOptions::MemoryOption>(mem, byteSize, dataVector.data());

            for (auto x : dataVector)
                out(x);
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(std::vector<HostType> &hostMem, SerialType snapshotField, ConvertOptions options)
        {
            hostMem.resize(snapshotField.size());

            int i = 0;
            for (const auto &x : snapshotField)
            {
                HostType temp;
                options.SubConverterOpts.Converter.Deserialize(temp, x);
                hostMem[i++] = temp;
            }
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(cl_mem_wrapper<HostType> mem, SerialType snapshotField, ConvertOptions options)
        {
            std::vector<HostType> hostVector;
            this->DeserializeBase(hostVector, snapshotField, options);

            const auto byteSize = hostVector.size() * sizeof(HostType);
            CLWriteBuffer<ConvertOptions::MemoryOption>(mem, byteSize, hostVector.data());
        }
    };

    RegisterCudaConverter1(VectorConverter);
} // namespace midas::opencl::protobuf