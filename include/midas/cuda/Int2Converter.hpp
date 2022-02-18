//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/Converters.hpp>

#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/CudaHelpers.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Int2Converter : public IConverter<Int2Converter>
    {
      public:
        template <typename ConvertOptions>
        void SerializeBase(const int2 *in, protobuf_support::pb_int2 *out, ConvertOptions convertOptions)
        {
            int2 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(int2), &result);

            out->set_x(result.x);
            out->set_y(result.y);
        }

        template <typename ConvertOptions>
        void DeserializeBase(int2 *out, const protobuf_support::pb_int2 &in, ConvertOptions convertOptions)
        {
            int2 result = {(int)in.x(), (int)in.y()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(int2), &result);
        }

        template <typename ConvertOptions>
        void DeserializeBase(int2 **out, const protobuf_support::pb_int2 &in, ConvertOptions convertOptions)
        {
            int2 result = {(int)in.x(), (int)in.y()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(int2), &result);
        }
    };

    RegisterCudaConverter1(Int2Converter);
} // namespace midas::cuda::protobuf
