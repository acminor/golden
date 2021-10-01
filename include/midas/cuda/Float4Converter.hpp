//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_FLOAT4CONVERTER_HPP
#define GOLDEN_MIDAS_FLOAT4CONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Float4Converter : public IConverter<Float4Converter>
    {
      public:
        template <typename ConvertOptions>
        void SerializeBase(const float4 *in, protobuf_support::pb_float4 *out, ConvertOptions convertOptions)
        {
            float4 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(float4), &result);

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
            out->set_w(result.w);
        }

        template <typename ConvertOptions>
        void DeserializeBase(float4 *out, const protobuf_support::pb_float4 &in, ConvertOptions convertOptions)
        {
            float4 result = {in.x(), in.y(), in.z(), in.w()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(float4), &result);
        }

        template <typename ConvertOptions>
        void DeserializeBase(float4 **out, const protobuf_support::pb_float4 &in, ConvertOptions convertOptions)
        {
            float4 result = {in.x(), in.y(), in.z(), in.w()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(float4), &result);
        }
    };

    RegisterCudaConverter1(Float4Converter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_FLOAT4CONVERTER_HPP
