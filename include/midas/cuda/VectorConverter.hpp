//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_VECTORCONVERTER_HPP
#define GOLDEN_MIDAS_VECTORCONVERTER_HPP

#include <midas/ProtobufUtility.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <optional>
#include <type_traits>
#include <vector>

namespace midas::cuda::protobuf
{
    using namespace midas::protobuf;

    class VectorConverter : public IConverter<VectorConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const std::vector<HostType> &in, SerialOutFunction out, ConvertOptions options)
        {
            static_assert(ConvertOptions::MemoryOption == MemoryOptions::Host,
                          "std::vector support is only for host memory.");
            for (auto x : in)
                out(x);
        }

        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(const std::pair<HostType *, size_t> in, SerialOutFunction out, ConvertOptions options)
        {
            const auto data = in.first;
            const auto length = in.second;
            std::vector<HostType> dataVector(length);
            const auto byteSize = length * sizeof(HostType);

            CudaReadBuffer<ConvertOptions::MemoryOption>(data, byteSize, dataVector.data());

            for (auto x : dataVector)
                out(x);
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(std::vector<HostType> &in, SerialType snapshotField, ConvertOptions options)
        {
            static_assert(ConvertOptions::MemoryOption == MemoryOptions::Host,
                          "std::vector support is only for host memory.");

            const auto length = snapshotField.size();
            in.resize(length);

            int i = 0;
            for (const auto &x : snapshotField)
            {
                HostType temp;
                options.SubConverterOpts.Converter.Deserialize(&temp, x);
                in[i++] = temp;
            }
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(HostType *hostMem, SerialType snapshotField, ConvertOptions options)
        {
            std::vector<HostType> hostVector;
            this->DeserializeBase(hostVector, snapshotField, make_options<MemoryOptions::Host>(options));

            const auto byteSize = hostVector.size() * sizeof(HostType);
            CudaWriteBuffer<ConvertOptions::MemoryOption>(hostMem, byteSize, hostVector.data());
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(HostType **hostMem, SerialType snapshotField, ConvertOptions options)
        {
            std::vector<HostType> hostVector;
            this->DeserializeBase(hostVector, snapshotField, make_options<MemoryOptions::Host>(options));

            const auto byteSize = hostVector.size() * sizeof(HostType);
            CudaWriteBuffer<ConvertOptions::MemoryOption>(hostMem, byteSize, hostVector.data());
        }
    };

    RegisterCudaConverter1(VectorConverter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_VECTORCONVERTER_HPP
