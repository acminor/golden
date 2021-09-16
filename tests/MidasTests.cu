//
// Created by acminor on 9/7/21.
//

#include <gtest/gtest.h>
#include <midas/Converters.hpp>
#include <midas/cuda/CudaConverters.hpp>

#include <iostream>

TEST(example, example)
{
    ExampleConverter a;
    midas::cuda::protobuf::Dim3Converter b;
    protobuf_support::dim3 out;
    dim3 in = {0, 1, 0};
    b.Serialize(in, out, CudaConvertOptions<CudaMemoryOptions::Host>{});

    EXPECT_THROW(a.Serialize(0, 0, CudaConvertOptions<CudaMemoryOptions::Host>{}), size_t);
    EXPECT_THROW(a.Serialize(0, 0, CudaConvertOptions<CudaMemoryOptions::Device>{}), float);

    midas::cuda::protobuf::VectorConverter VectorConverter;
    std::vector data = {1, 2, 3};
    protobuf_support::float_vec outVec;
    VectorConverter.Serialize(
        data, [&](const auto &x) { outVec.add_data(x); },
        midas::cuda::protobuf::VectorConverterOptions<CudaMemoryOptions::Host>());

    std::cout << outVec.data(0) << std::endl;
    std::cout << outVec.data(1) << std::endl;
    std::cout << outVec.data(2) << std::endl;

    VectorConverter.Serialize(
        data, [&](const auto &x) { outVec.add_data(x); },
        midas::cuda::protobuf::VectorConverterOptions<CudaMemoryOptions::Host>());
}