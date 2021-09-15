//
// Created by acminor on 9/8/21.
//

#ifndef GOLDEN_PRIMITIVECONVERTER_HPP
#define GOLDEN_PRIMITIVECONVERTER_HPP

#include <midas/ProtobufSupport.pb.h>
#include <midas/RegisterCudaConverter.hpp>
#include <vector_types.h>

#include <cuda.h>

namespace midas::cuda::protobuf
{
    class PrimitiveConverter : public IConverter<PrimitiveConverter>
    {
      public:
        template <typename HostType, typename SerialOutFunction,
                  typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const HostType &in, SerialOutFunction out, ConvertOptions convertOptions = {})
        {
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            HostType result;
            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                result = in;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&result, &in, sizeof(HostType), cudaMemcpyDeviceToHost);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol(&result, in, sizeof(HostType));
            }

            out(result);
        }

        template <typename HostType, typename SerialType,
                  typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void DeserializeBase(HostType &out, const SerialType &in, ConvertOptions convertOptions = {})
        {
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            HostType result = in;

            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                out = result;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&out, &result, sizeof(HostType), cudaMemcpyHostToDevice);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyToSymbol(out, &result, sizeof(HostType));
            }
        }
    };

    RegisterCudaConverter1(PrimitiveConverter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_PRIMITIVECONVERTER_HPP
