//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_INT4CONVERTER_HPP
#define GOLDEN_MIDAS_INT4CONVERTER_HPP

#include <midas/Converters.hpp>

#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/CudaHelpers.hpp>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Int4Converter : public IConverter<Int4Converter>
    {
      public:
        template <typename ConvertOptions>
        void SerializeBase(const int4 *in, protobuf_support::pb_int4 *out, ConvertOptions convertOptions)
        {
            int4 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(int4), &result);

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
            out->set_w(result.w);
        }

        template <typename ConvertOptions>
        void DeserializeBase(int4 *out, const protobuf_support::pb_int4 &in, ConvertOptions convertOptions)
        {
            int4 result = {(int)in.x(), (int)in.y(), (int)in.z(), (int)in.w()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(int4), &result);
        }

        template <typename ConvertOptions>
        void DeserializeBase(int4 **out, const protobuf_support::pb_int4 &in, ConvertOptions convertOptions)
        {
            int4 result = {(int)in.x(), (int)in.y(), (int)in.z(), (int)in.w()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(int4), &result);
        }
    };

    RegisterCudaConverter1(Int4Converter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_MIDAS_INT4CONVERTER_HPP
