//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_FLOAT4CONVERTER_HPP
#define GOLDEN_FLOAT4CONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/RegisterCudaConverter.hpp>
#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Float4Converter : public IConverter<Float4Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const float4 &in, protobuf_support::float4 &out, ConvertOptions convertOptions = {})
        {
            /*
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            float4 result;
            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                result = in;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&result, &in, sizeof(float4), cudaMemcpyDeviceToHost);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol(&result, &in, sizeof(float4), cudaMemcpyDeviceToHost);
            }

            out.set_x(result.x);
            out.set_y(result.y);
            out.set_z(result.z);
            out.set_w(result.w);
            */
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const float4 &in, protobuf_support::float4 *out, ConvertOptions convertOptions = {})
        {
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            float4 result;
            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                result = in;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&result, &in, sizeof(float4), cudaMemcpyDeviceToHost);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol(&result, in, sizeof(float4));
            }

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
            out->set_w(result.w);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void DeserializeBase(float4 &out, const protobuf_support::float4 &in, ConvertOptions convertOptions = {})
        {
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            float4 result = {in.x(), in.y(), in.z(), in.w()};

            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                out = result;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&out, &result, sizeof(float4), cudaMemcpyHostToDevice);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyToSymbol(out, &result, sizeof(float4));
            }
        }
    };

    RegisterCudaConverter1(Float4Converter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_FLOAT4CONVERTER_HPP
