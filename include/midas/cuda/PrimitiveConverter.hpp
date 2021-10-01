//
// Created by acminor on 9/8/21.
//

#ifndef GOLDEN_MIDAS_PRIMITIVECONVERTER_HPP
#define GOLDEN_MIDAS_PRIMITIVECONVERTER_HPP

#include <midas/Converters.hpp>

#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/CudaHelpers.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

/**
 * @namespace midas::cuda::protobuf
 */
namespace midas::cuda::protobuf
{
    /**
     * @class PrimitiveConverter
     * @brief Something
     */
    class PrimitiveConverter : public IConverter<PrimitiveConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction,
                  typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const HostType *in, SerialOutFunction out, ConvertOptions convertOptions = {})
        {
            HostType result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(HostType), &result);
            out(result);
        }

        /**
         * @note both a HostType& and HostType** are needed
         *       - the HostType& is for host/symbol side non-allocated memory (globals and stack variables)
         *       - the HostType** is for everything else
         */
        template <typename HostType, typename SerialType,
                  typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void DeserializeBase(HostType *out, const SerialType &in, ConvertOptions convertOptions = {})
        {
            // assures that the serial type can be implicitly converted to the HostType
            HostType result = in;
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(HostType), &result);
        }

        template <typename HostType, typename SerialType,
                  typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void DeserializeBase(HostType **out, const SerialType &in, ConvertOptions convertOptions = {})
        {
            // assures that the serial type can be implicitly converted to the HostType
            HostType result = in;
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(HostType), &result);
        }
    };

    RegisterCudaConverter1(PrimitiveConverter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_PRIMITIVECONVERTER_HPP
