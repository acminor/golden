//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MALLOCINTEROPARRAYCONVERTER_HPP
#define GOLDEN_MALLOCINTEROPARRAYCONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/RegisterCudaConverter.hpp>

size_t cudaGetArraySize(void *);
namespace midas::cuda::protobuf
{
    class MallocInteropArrayConverter : public IConverter<MallocInteropArrayConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction, typename ConvertOptions>
        void SerializeBase(HostType *in, SerialOutFunction out, ConvertOptions options)
        {
            auto size = cudaGetArraySize((void *)in) / sizeof(HostType);

            if (size != 0)
            {
                // auto vectorData = cudaToVector(data, size);
                // cudaToProtobuf(vectorData, updateProtobufElement);
                midas::cuda::protobuf::converters::VectorConverter.Serialize(std::make_pair(in, size), out, options);
            }
            else
            {
                throw "Ahhh, using memory buffer without size and not tracked";
            }
        }

        template <typename HostType, typename SerialType, typename ConvertOptions>
        void DeserializeBase(HostType, SerialType, ConvertOptions)
        {
            throw "Not implemented";
        }
    };

    RegisterCudaConverter1(MallocInteropArrayConverter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MALLOCINTEROPARRAYCONVERTER_HPP
