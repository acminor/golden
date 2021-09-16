//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_VECTORCONVERTER_HPP
#define GOLDEN_MIDAS_VECTORCONVERTER_HPP

#include <midas/ProtobufUtility.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <optional>
#include <type_traits>

namespace midas::cuda::protobuf
{
    using namespace midas::protobuf;

    class NoAdditionalConverter
    {
      public:
        template <typename HostType, typename SerialType>
        void Serialize(const HostType &in, const SerialType &out) const
        {
        }

        template <typename HostType, typename SerialType>
        void Deserialize(const HostType &in, const SerialType &out) const
        {
        }

        template <typename HostType, typename SerialType,
                  std::enable_if_t<std::is_convertible_v<SerialType, HostType>, bool> = true>
        void Serialize(const HostType &in, SerialType &out)
        {
            out = (HostType)in;
        }

        template <typename HostType, typename SerialType,
                  std::enable_if_t<std::is_convertible_v<SerialType, HostType>, bool> = true>
        void Deserialize(HostType &out, const SerialType &in)
        {
            out = (HostType)in;
        }
    };

    class VectorConverter;
    template <CudaMemoryOptions MemoryOptions, typename FilledConverter = NoAdditionalConverter>
    class VectorConverterOptions : CudaConvertOptions<MemoryOptions>
    {
      public:
        using Tag = VectorConverterOptions<CudaMemoryOptions::Host, void>;

        VectorConverterOptions() : m_converter(FilledConverter{})
        {
        }

        VectorConverterOptions(FilledConverter converter) : m_converter(converter)
        {
        }

        friend VectorConverter;

      private:
        FilledConverter m_converter;
    };

    template <CudaMemoryOptions MemoryOptions, typename FilledConverter = NoAdditionalConverter>
    VectorConverterOptions<MemoryOptions, FilledConverter> makeVectorConverterOptions(FilledConverter converter = {})
    {
        return VectorConverterOptions<MemoryOptions, FilledConverter>(converter);
    }

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
        void SerializeBase(const std::pair<HostType *, size_t> in, SerialOutFunction out, ConvertOptions options)
        {
            auto data = in.first;
            auto length = in.second;

            auto dataVector = std::vector<HostType>(length);

            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                dataVector.insert(data, data + length);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy((void *)dataVector.data(), (void *)data, length * sizeof(HostType), cudaMemcpyDeviceToHost);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol((void *)dataVector.data(), (void *)data, length * sizeof(HostType),
                                     cudaMemcpyDeviceToHost);
            }

            for (auto x : dataVector)
                out(x);
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(HostType **hostMem, SerialType snapshotField, ConvertOptions options)
        {
            std::vector<HostType> host_vector(snapshotField.size());
            const auto byteSize = host_vector.size() * sizeof(HostType);

            int i = 0;
            for (const auto &x : snapshotField)
            {
                HostType temp;
                options.m_converter.Deserialize(temp, x);
                host_vector[i++] = temp;
            }

            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                *hostMem = malloc(byteSize);
                memcpy(*hostMem, host_vector.data(), byteSize);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMalloc((void **)hostMem, byteSize);
                cudaMemcpy((void *)*hostMem, (void *)host_vector.data(), byteSize, cudaMemcpyHostToDevice);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMalloc((void **)hostMem, byteSize);
                cudaMemcpyToSymbol((void *)*hostMem, (void *)host_vector.data(), byteSize, cudaMemcpyHostToDevice);
            }
        }
    };

    RegisterCudaConverter1(VectorConverter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_VECTORCONVERTER_HPP
