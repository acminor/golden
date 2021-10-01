//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_DIM3CONVERTER_HPP
#define GOLDEN_MIDAS_DIM3CONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/CudaHelpers.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Dim3Converter : public IConverter<Dim3Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void SerializeBase(const dim3 *in, protobuf_support::pb_dim3 *out, ConvertOptions convertOptions)
        {
            dim3 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(dim3), &result);

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
        }

        template <typename ConvertOptions>
        void DeserializeBase(dim3 *out, const protobuf_support::pb_dim3 &in, ConvertOptions convertOptions)
        {
            static_assert(IsCopyOnlyOption(ConvertOptions::MemoryOption));
            dim3 result = {(unsigned int)in.x(), (unsigned int)in.y(), (unsigned int)in.z()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(dim3), &result);
        }

        template <typename ConvertOptions = CudaConvertOptions<MemoryOptions::Host>>
        void DeserializeBase(dim3 **out, const protobuf_support::pb_dim3 &in, ConvertOptions convertOptions = {})
        {
            dim3 result = {(unsigned int)in.x(), (unsigned int)in.y(), (unsigned int)in.z()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(dim3), &result);
        }
    };

    RegisterCudaConverter1(Dim3Converter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_DIM3CONVERTER_HPP
