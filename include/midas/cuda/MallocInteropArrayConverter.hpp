//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_MALLOCINTEROPARRAYCONVERTER_HPP
#define GOLDEN_MIDAS_MALLOCINTEROPARRAYCONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

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

#endif // GOLDEN_MIDAS_MALLOCINTEROPARRAYCONVERTER_HPP
