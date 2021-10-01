//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/cuda/Dim3Converter.hpp>

#include <MidasTests.pb.h>

TEST(Dim3Tests, Serialize)
{
    auto Converter = midas::cuda::protobuf::converters::Dim3Converter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::Dim3Wrapper serialExpected;
    {
        auto data = serialExpected.mutable_data();
        data->set_x(1);
        data->set_y(2);
        data->set_z(3);
    }

    dim3 hostIn = {1, 2, 3};

    midas_tests::Dim3Wrapper serialOut;
    Converter.Serialize(&hostIn, serialOut.mutable_data(), make_options<MemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    dim3 *deviceIn;
    cudaMalloc((void **)&deviceIn, sizeof(hostIn));
    cudaMemcpy(deviceIn, &hostIn, sizeof(hostIn), cudaMemcpyHostToDevice);
    Converter.Serialize(deviceIn, serialOut.mutable_data(), make_options<MemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(Dim3Tests, Deserialize)
{
    auto Converter = midas::cuda::protobuf::converters::Dim3Converter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    midas_tests::Dim3Wrapper serialIn;
    {
        auto data = serialIn.mutable_data();
        data->set_x(1);
        data->set_y(2);
        data->set_z(3);
    }

    dim3 hostExpected = {1, 2, 3};

    dim3 hostOut;
    Converter.Deserialize(&hostOut, serialIn.data(), make_options<MemoryOptions::Host>());
    ASSERT_EQ(hostOut.x, hostExpected.x);
    ASSERT_EQ(hostOut.y, hostExpected.y);
    ASSERT_EQ(hostOut.z, hostExpected.z);

    dim3 *deviceOut;
    Converter.Deserialize(&deviceOut, serialIn.data(), make_options<MemoryOptions::Device>());
    cudaMemcpy(&hostOut, deviceOut, sizeof(hostOut), cudaMemcpyDeviceToHost);
    ASSERT_EQ(hostOut.x, hostExpected.x);
    ASSERT_EQ(hostOut.y, hostExpected.y);
    ASSERT_EQ(hostOut.z, hostExpected.z);
}