//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_FLOAT2CONVERTER_HPP
#define GOLDEN_MIDAS_FLOAT2CONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Float2Converter : public IConverter<Float2Converter>
    {
      public:
        template <typename ConvertOptions>
        void SerializeBase(const float2 *in, protobuf_support::pb_float2 *out, ConvertOptions convertOptions)
        {
            float2 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(float2), &result);

            out->set_x(result.x);
            out->set_y(result.y);
        }

        template <typename ConvertOptions>
        void DeserializeBase(float2 *out, const protobuf_support::pb_float2 &in, ConvertOptions convertOptions)
        {
            float2 result = {in.x(), in.y()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(float2), &result);
        }

        template <typename ConvertOptions>
        void DeserializeBase(float2 **out, const protobuf_support::pb_float2 &in, ConvertOptions convertOptions)
        {
            float2 result = {in.x(), in.y()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(float2), &result);
        }
    };

    RegisterCudaConverter1(Float2Converter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_FLOAT2CONVERTER_HPP
