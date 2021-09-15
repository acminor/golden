//
// Created by acminor on 9/7/21.
//

#include <gtest/gtest.h>
#include <midas/Converters.hpp>
#include <midas/CudaConverters.hpp>
#include <midas/cuda.hpp>

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
    // a.Serialize(0, 0, ConvertOptions{});
    // a.Serialize(0, 0, CudaConvertOptions<2>{});

    midas::cuda::protobuf::VectorConverter VectorConverter;
    std::vector data = {1, 2, 3};
    protobuf_support::float_vec outVec;
    auto options = midas::cuda::protobuf::makeVectorConverterOptions<CudaMemoryOptions::Host>(
        [&](const auto &x) { outVec.add_data(x); });
    VectorConverter.Serialize(data, outVec, options);

    std::cout << outVec.data(0) << std::endl;
    std::cout << outVec.data(1) << std::endl;
    std::cout << outVec.data(2) << std::endl;

    VectorConverter.Serialize(data, outVec,
                              midas::cuda::protobuf::makeVectorConverterOptions<CudaMemoryOptions::Host>(
                                  [&](const auto &x) { outVec.add_data(x); }));

    auto filledConverterOptions = midas::cuda::protobuf::makeVectorConverterOptions<CudaMemoryOptions::Host>(
        [&](const auto &x) { outVec.add_data(x); },
        FilledConverter(midas::cuda::protobuf::Dim3Converter{}, CudaConvertOptions<CudaMemoryOptions::Host>{}));
}