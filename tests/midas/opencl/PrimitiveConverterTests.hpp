//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>
#include <midas/opencl/PrimitiveConverter.hpp>

#include <MidasTests.pb.h>
#include <opencl/TestsUtility.hpp>

TEST(PrimitiveTests, Serialize)
{
    auto Converter = midas::opencl::protobuf::converters::PrimitiveConverter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::Primatives serialExpected;
    serialExpected.set_f(3.5);
    serialExpected.set_i(3);

    midas_tests::Primatives serialOut;
    float hostFloatIn = 3.5;
    long hostLongIn = 3;
    Converter.Serialize(
        hostFloatIn, [&](const auto x) { serialOut.set_f(x); }, make_options<CudaMemoryOptions::Host>());
    Converter.Serialize(
        hostLongIn, [&](const auto x) { serialOut.set_i(x); }, make_options<CudaMemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    auto deviceFloatIn = easyBufferCreate(&hostFloatIn, sizeof(hostFloatIn));
    auto deviceLongIn = easyBufferCreate(&hostLongIn, sizeof(hostLongIn));
    Converter.Serialize(
        cl_mem_wrapper<decltype(hostFloatIn)>(deviceFloatIn, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
        [&](const auto x) { serialOut.set_f(x); }, make_options<CudaMemoryOptions::Device>());
    Converter.Serialize(
        cl_mem_wrapper<decltype(hostLongIn)>(deviceLongIn, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
        [&](const auto x) { serialOut.set_i(x); }, make_options<CudaMemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(PrimitiveTests, Deserialize)
{
    auto Converter = midas::opencl::protobuf::converters::PrimitiveConverter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    midas_tests::Primatives serialIn;
    serialIn.set_i(3);
    serialIn.set_f(3.5);

    float hostFloatExpected = 3.5;
    float hostLongExpected = 3;

    float hostFloatOut = 3.5;
    long hostLongOut = 3;

    Converter.Deserialize(hostFloatOut, serialIn.f(), make_options<CudaMemoryOptions::Host>());
    Converter.Deserialize(hostLongOut, serialIn.i(), make_options<CudaMemoryOptions::Host>());

    ASSERT_EQ(hostFloatOut, hostFloatExpected);
    ASSERT_EQ(hostLongOut, hostLongExpected);

    float hostFloatZero = 0.0;
    long hostLongZero = 0;
    auto deviceFloatOut = easyBufferCreate(&hostFloatZero, sizeof(hostFloatZero));
    auto deviceLongOut = easyBufferCreate(&hostLongZero, sizeof(hostLongZero));
    Converter.Deserialize(
        cl_mem_wrapper<decltype(hostLongZero)>(deviceLongOut, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
        serialIn.i(), make_options<CudaMemoryOptions::Device>());
    Converter.Deserialize(cl_mem_wrapper<decltype(hostFloatZero)>(deviceFloatOut, CL_MEM_READ_WRITE, OpenClData.context,
                                                                  OpenClData.queue),
                          serialIn.f(), make_options<CudaMemoryOptions::Device>());

    easyBufferRead(deviceFloatOut, &hostFloatOut, sizeof(hostFloatOut));
    easyBufferRead(deviceLongOut, &hostLongOut, sizeof(hostLongOut));

    ASSERT_EQ(hostFloatOut, hostFloatExpected);
    ASSERT_EQ(hostLongOut, hostLongExpected);
}