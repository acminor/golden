//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/cuda/Float4Converter.hpp>
#include <midas/cuda/VectorConverter.hpp>

#include <MidasTests.pb.h>

TEST(VectorTests, SerializePrimitive)
{
    auto Converter = midas::cuda::protobuf::converters::VectorConverter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::IntArray serialExpected;
    {
        for (int i = 0; i < 10; i++)
            serialExpected.add_data(i);
    }

    std::vector<int> hostIn(10);
    for (int i = 0; i < 10; i++)
        hostIn[i] = i;

    midas_tests::IntArray serialOut;
    Converter.Serialize(
        hostIn, [&](const auto &x) { serialOut.add_data(x); }, make_options<MemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    serialOut.clear_data();
    int *deviceIn;
    cudaMalloc((void **)&deviceIn, 10 * sizeof(hostIn[0]));
    cudaMemcpy(deviceIn, hostIn.data(), 10 * sizeof(hostIn[0]), cudaMemcpyHostToDevice);
    Converter.Serialize(
        std::make_pair(deviceIn, 10UL), [&](const auto &x) { serialOut.add_data(x); },
        make_options<MemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(VectorTests, DeserializePrimitive)
{
    auto Converter = midas::cuda::protobuf::converters::VectorConverter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    midas_tests::IntArray serialIn;
    {
        for (int i = 0; i < 10; i++)
            serialIn.add_data(i);
    }

    std::vector<int> hostExpected(10);
    for (int i = 0; i < 10; i++)
        hostExpected[i] = i;

    std::vector<int> hostOut;
    Converter.Deserialize(hostOut, serialIn.data(), make_options<MemoryOptions::Host>());
    ASSERT_EQ(hostOut, hostExpected);

    int *deviceOut;
    Converter.Deserialize(&deviceOut, serialIn.data(), make_options<MemoryOptions::Device>());
    cudaMemcpy(hostOut.data(), deviceOut, 10 * sizeof(hostOut[0]), cudaMemcpyHostToDevice);
    ASSERT_EQ(hostOut, hostExpected);
}

TEST(VectorTests, DeserializeFloat4)
{
    auto Converter = midas::cuda::protobuf::converters::VectorConverter;

    using namespace midas::cuda::protobuf;
    using namespace midas::cuda;

    midas_tests::Float4Array serialIn;
    {
        for (int i = 0; i < 10; i++)
        {
            auto data = serialIn.add_data();
            data->set_x(1.5);
            data->set_y(2.0);
            data->set_z(3.5);
            data->set_w(4.0);
        }
    }

    std::vector<float4> hostExpected(10);
    for (int i = 0; i < 10; i++)
        hostExpected[i] = {1.5, 2.0, 3.5, 4.0};

    std::vector<float4> hostOut;
    Converter.Deserialize(hostOut, serialIn.data(),
                          make_options<MemoryOptions::Host>(
                              FilledConverter(converters::Float4Converter, make_options<MemoryOptions::Host>())));
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(hostOut[i].x, hostExpected[i].x);
        ASSERT_EQ(hostOut[i].y, hostExpected[i].y);
        ASSERT_EQ(hostOut[i].z, hostExpected[i].z);
        ASSERT_EQ(hostOut[i].w, hostExpected[i].w);
    }

    float4 *deviceOut;
    Converter.Deserialize(&deviceOut, serialIn.data(),
                          make_options<MemoryOptions::Device>(
                              FilledConverter(converters::Float4Converter, make_options<MemoryOptions::Host>())));
    cudaMemcpy(hostOut.data(), deviceOut, 10 * sizeof(hostOut[0]), cudaMemcpyHostToDevice);
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(hostOut[i].x, hostExpected[i].x);
        ASSERT_EQ(hostOut[i].y, hostExpected[i].y);
        ASSERT_EQ(hostOut[i].z, hostExpected[i].z);
        ASSERT_EQ(hostOut[i].w, hostExpected[i].w);
    }
}