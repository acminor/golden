//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/opencl/Int4Converter.hpp>
#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>

#include <MidasTests.pb.h>
#include <opencl/TestsUtility.hpp>

TEST(Int4Tests, Serialize)
{
    auto Converter = midas::opencl::protobuf::converters::Int4Converter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::Int4Wrapper serialExpected;
    {
        auto data = serialExpected.mutable_data();
        data->set_x(1);
        data->set_y(2);
        data->set_z(3);
        data->set_w(4);
    }

    cl_int4 hostIn;
    hostIn.x = 1;
    hostIn.y = 2;
    hostIn.z = 3;
    hostIn.w = 4;

    midas_tests::Int4Wrapper serialOut;
    Converter.Serialize(hostIn, serialOut.mutable_data(), make_options<CudaMemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    auto deviceIn = easyBufferCreate(&hostIn, sizeof(hostIn));
    Converter.Serialize(
        cl_mem_wrapper<decltype(hostIn)>(deviceIn, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
        serialOut.mutable_data(), make_options<CudaMemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(Int4Tests, Deserialize)
{
    InitializeOpenCL();

    auto Converter = midas::opencl::protobuf::converters::Int4Converter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    midas_tests::Int4Wrapper serialIn;
    {
        auto data = serialIn.mutable_data();
        data->set_x(1);
        data->set_y(2);
        data->set_z(3);
        data->set_w(4);
    }

    cl_int4 hostExpected;
    hostExpected.x = 1;
    hostExpected.y = 2;
    hostExpected.z = 3;
    hostExpected.w = 4;

    cl_int4 hostOut;
    Converter.Deserialize(hostOut, serialIn.data(), make_options<CudaMemoryOptions::Host>());

    ASSERT_EQ(hostOut.x, hostExpected.x);
    ASSERT_EQ(hostOut.y, hostExpected.y);
    ASSERT_EQ(hostOut.z, hostExpected.z);
    ASSERT_EQ(hostOut.w, hostExpected.w);

    cl_mem deviceOut;
    Converter.Deserialize(
        cl_mem_wrapper<decltype(hostOut)>(deviceOut, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
        serialIn.data(), make_options<CudaMemoryOptions::Device>());

    easyBufferRead(deviceOut, &hostOut, sizeof(hostOut));

    ASSERT_EQ(hostOut.x, hostExpected.x);
    ASSERT_EQ(hostOut.y, hostExpected.y);
    ASSERT_EQ(hostOut.z, hostExpected.z);
    ASSERT_EQ(hostOut.w, hostExpected.w);
}