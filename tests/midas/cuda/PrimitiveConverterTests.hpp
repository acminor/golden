//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/cuda/PrimitiveConverter.hpp>

#include <MidasTests.pb.h>

TEST(PrimitiveTests, Serialize)
{
    auto Converter = midas::cuda::protobuf::converters::PrimitiveConverter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::Primatives serialExpected;
    serialExpected.set_f(3.5);
    serialExpected.set_i(3);

    midas_tests::Primatives serialOut;
    float hostFloatIn = 3.5;
    long hostLongIn = 3;
    Converter.Serialize(
        &hostFloatIn, [&](const auto x) { serialOut.set_f(x); }, make_options<MemoryOptions::Host>());
    Converter.Serialize(
        &hostLongIn, [&](const auto x) { serialOut.set_i(x); }, make_options<MemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    float *deviceFloatIn;
    long *deviceLongIn;
    cudaMalloc((void **)&deviceFloatIn, sizeof(hostFloatIn));
    cudaMemcpy(deviceFloatIn, &hostFloatIn, sizeof(hostFloatIn), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&deviceLongIn, sizeof(hostLongIn));
    cudaMemcpy(deviceLongIn, &hostLongIn, sizeof(hostLongIn), cudaMemcpyHostToDevice);
    Converter.Serialize(
        deviceFloatIn, [&](const auto x) { serialOut.set_f(x); }, make_options<MemoryOptions::Device>());
    Converter.Serialize(
        deviceLongIn, [&](const auto x) { serialOut.set_i(x); }, make_options<MemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(PrimitiveTests, Deserialize)
{
    auto Converter = midas::cuda::protobuf::converters::PrimitiveConverter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    midas_tests::Primatives serialIn;
    serialIn.set_i(3);
    serialIn.set_f(3.5);

    float hostFloatExpected = 3.5;
    float hostLongExpected = 3;

    float hostFloatOut;
    long hostLongOut;

    Converter.Deserialize(&hostFloatOut, serialIn.f(), make_options<MemoryOptions::Host>());
    Converter.Deserialize(&hostLongOut, serialIn.i(), make_options<MemoryOptions::Host>());

    ASSERT_EQ(hostFloatOut, hostFloatExpected);
    ASSERT_EQ(hostLongOut, hostLongExpected);

    float *deviceFloatOut;
    long *deviceLongOut;
    Converter.Deserialize(&deviceLongOut, serialIn.i(), make_options<MemoryOptions::Device>());
    Converter.Deserialize(&deviceFloatOut, serialIn.f(), make_options<MemoryOptions::Device>());
    cudaMemcpy(&hostLongOut, deviceLongOut, sizeof(hostLongOut), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostFloatOut, deviceFloatOut, sizeof(hostFloatOut), cudaMemcpyDeviceToHost);
    ASSERT_EQ(hostFloatOut, hostFloatExpected);
    ASSERT_EQ(hostLongOut, hostLongExpected);
}