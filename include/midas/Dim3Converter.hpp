//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_DIM3CONVERTER_HPP
#define GOLDEN_DIM3CONVERTER_HPP

#include <midas/Converters.hpp>
#include <midas/ProtobufSupport.pb.h>
#include <midas/RegisterCudaConverter.hpp>
#include <vector_types.h>

namespace midas::cuda::protobuf
{
    class Dim3Converter : public IConverter<Dim3Converter>
    {
      public:
        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const dim3 &in, protobuf_support::dim3 &out, ConvertOptions convertOptions = {})
        {
            /*
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            dim3 result;
            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                result = in;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&result, &in, sizeof(dim3), cudaMemcpyDeviceToHost);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol(&result, &in, sizeof(dim3), cudaMemcpyDeviceToHost);
            }

            out.set_x(result.x);
            out.set_y(result.y);
            out.set_z(result.z);
            */
            this->SerializeBase(in, &out, convertOptions);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void SerializeBase(const dim3 &in, protobuf_support::dim3 *out, ConvertOptions convertOptions = {})
        {
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            dim3 result;
            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                result = in;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&result, &in, sizeof(dim3), cudaMemcpyDeviceToHost);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol(&result, &in, sizeof(dim3), cudaMemcpyDeviceToHost);
            }

            out->set_x(result.x);
            out->set_y(result.y);
            out->set_z(result.z);
        }

        template <typename ConvertOptions = CudaConvertOptions<CudaMemoryOptions::Host>>
        void DeserializeBase(dim3 &out, const protobuf_support::dim3 &in, ConvertOptions convertOptions = {})
        {
            static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

            dim3 result = {(unsigned int)in.x(), (unsigned int)in.y(), (unsigned int)in.z()};

            if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            {
                out = result;
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            {
                cudaMemcpy(&out, &result, sizeof(dim3), cudaMemcpyHostToDevice);
            }
            else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Symbol)
            {
                cudaMemcpyFromSymbol(&out, &result, sizeof(dim3), cudaMemcpyHostToDevice);
            }
        }
    };

    RegisterCudaConverter1(Dim3Converter);
} // namespace midas::cuda::protobuf

#endif // GOLDEN_DIM3CONVERTER_HPP
