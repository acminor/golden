//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Float3Converter : public IConverter<Float3Converter>
    {
      public:
        template <typename ConvertOptions>
        void SerializeBase(const float3 *in, protobuf_support::pb_float3 *out, ConvertOptions convertOptions)
        {
            float3 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(float3), &result);

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
        }

        template <typename ConvertOptions>
        void DeserializeBase(float3 *out, const protobuf_support::pb_float3 &in, ConvertOptions convertOptions)
        {
            float3 result = {in.x(), in.y(), in.z()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(float3), &result);
        }

        template <typename ConvertOptions>
        void DeserializeBase(float3 **out, const protobuf_support::pb_float3 &in, ConvertOptions convertOptions)
        {
            float3 result = {in.x(), in.y(), in.z()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(float3), &result);
        }
    };

    RegisterCudaConverter1(Float3Converter);
} // namespace midas::cuda::protobuf
