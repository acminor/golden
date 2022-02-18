//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/cuda/RegisterCudaConverter.hpp>

#include <vector_types.h>
#include <cuComplex.h>

namespace midas::cuda::protobuf
{
    class ComplexConverter : public IConverter<ComplexConverter>
    {
      public:
        template <typename ConvertOptions>
        void SerializeBase(const cuComplex *in, protobuf_support::pb_complex *out, ConvertOptions convertOptions)
        {
            float2 result;
            CudaReadBuffer<ConvertOptions::MemoryOption>(in, sizeof(cuComplex), &result);

            out->set_real(result.x);
            out->set_imag(result.y);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cuComplex *out, const protobuf_support::pb_complex &in, ConvertOptions convertOptions)
        {
            cuComplex result = {in.real(), in.imag()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(cuComplex), &result);
        }

        template <typename ConvertOptions>
        void DeserializeBase(cuComplex **out, const protobuf_support::pb_complex &in, ConvertOptions convertOptions)
        {
            cuComplex result = {in.real(), in.imag()};
            CudaWriteBuffer<ConvertOptions::MemoryOption>(out, sizeof(cuComplex), &result);
        }
    };

    RegisterCudaConverter1(ComplexConverter);
} // namespace midas::cuda::protobuf
