//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_TO_HPP
#define GOLDEN_TO_HPP

#include <type_traits>

#include "../utility.hpp"

namespace midas::cuda::protobuf
{
    template <typename T, typename Function, typename CudaMemoryType = midas::cuda::utility::CudaNormalMemory>
    void to(const T &data, Function updateProtobufElement)
    {
        auto val = midas::cuda::utility::cudaToHost<T, CudaMemoryType>(data);
        updateProtobufElement(val);
    }

#ifdef MIDAS_CUDA_MALLOC_INTEROP_ENABLED
    template <typename T, typename Function> void to(const T *data, Function updateProtobufElement)
    {
        auto size = cudaGetArraySize((void *)data) / sizeof(T);

        if (size != 0)
        {
            auto vectorData = cudaToVector(data, size);
            cudaToProtobuf(vectorData, updateProtobufElement);
        }
        else
        {
            throw "Ahhh, using memory buffer without size and not tracked";
        }
    }
#endif
} // namespace midas::cuda::protobuf

#endif // GOLDEN_TO_HPP
