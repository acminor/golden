//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/cuda/Float4Converter.hpp>

#include <MidasTests.pb.h>

TEST(Float4Tests, Serialize)
{
    auto Converter = midas::cuda::protobuf::converters::Float4Converter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::Float4Wrapper serialExpected;
    {
        auto data = serialExpected.mutable_data();
        data->set_x(1.5);
        data->set_y(2.0);
        data->set_z(3.5);
        data->set_w(4.0);
    }

    float4 hostIn;
    hostIn.x = 1.5;
    hostIn.y = 2.0;
    hostIn.z = 3.5;
    hostIn.w = 4.0;

    midas_tests::Float4Wrapper serialOut;
    Converter.Serialize(&hostIn, serialOut.mutable_data(), make_options<MemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    float4 *deviceIn;
    cudaMalloc((void **)&deviceIn, sizeof(hostIn));
    cudaMemcpy(deviceIn, &hostIn, sizeof(hostIn), cudaMemcpyHostToDevice);
    Converter.Serialize(deviceIn, serialOut.mutable_data(), make_options<MemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(Float4Tests, Deserialize)
{
    auto Converter = midas::cuda::protobuf::converters::Float4Converter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    midas_tests::Float4Wrapper serialIn;
    {
        auto data = serialIn.mutable_data();
        data->set_x(1.5);
        data->set_y(2.0);
        data->set_z(3.5);
        data->set_w(4.0);
    }

    float4 hostExpected;
    hostExpected.x = 1.5;
    hostExpected.y = 2.0;
    hostExpected.z = 3.5;
    hostExpected.w = 4.0;

    float4 hostOut;
    Converter.Deserialize(&hostOut, serialIn.data(), make_options<MemoryOptions::Host>());

    ASSERT_EQ(hostOut.x, hostExpected.x);
    ASSERT_EQ(hostOut.y, hostExpected.y);
    ASSERT_EQ(hostOut.z, hostExpected.z);
    ASSERT_EQ(hostOut.w, hostExpected.w);

    float4 *deviceOut;
    Converter.Deserialize(&deviceOut, serialIn.data(), make_options<MemoryOptions::Device>());
    cudaMemcpy(&hostOut, deviceOut, sizeof(hostOut), cudaMemcpyDeviceToHost);
    ASSERT_EQ(hostOut.x, hostExpected.x);
    ASSERT_EQ(hostOut.y, hostExpected.y);
    ASSERT_EQ(hostOut.z, hostExpected.z);
    ASSERT_EQ(hostOut.w, hostExpected.w);
}