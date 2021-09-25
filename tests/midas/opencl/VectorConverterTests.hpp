//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/opencl/Float4Converter.hpp>
#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>
#include <midas/opencl/VectorConverter.hpp>

#include <MidasTests.pb.h>
#include <opencl/TestsUtility.hpp>

TEST(VectorTests, SerializePrimitive)
{
    auto Converter = midas::opencl::protobuf::converters::VectorConverter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

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
        hostIn, [&](const auto &x) { serialOut.add_data(x); }, make_options<CudaMemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    serialOut.clear_data();
    auto deviceIn = easyBufferCreate(hostIn.data(), sizeof(int) * 10);
    Converter.Serialize(
        std::make_pair(cl_mem_wrapper<int>(deviceIn, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue), 10ul),
        [&](const auto &x) { serialOut.add_data(x); }, make_options<CudaMemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(VectorTests, DeserializePrimitive)
{
    auto Converter = midas::opencl::protobuf::converters::VectorConverter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    midas_tests::IntArray serialIn;
    {
        for (int i = 0; i < 10; i++)
            serialIn.add_data(i);
    }

    std::vector<int> hostExpected(10);
    for (int i = 0; i < 10; i++)
        hostExpected[i] = i;

    std::vector<int> hostOut;
    Converter.Deserialize(hostOut, serialIn.data(), make_options<CudaMemoryOptions::Host>());
    ASSERT_EQ(hostOut, hostExpected);

    std::vector<int> hostZeros(10);
    for (int i = 0; i < 10; i++)
        hostZeros[i] = 0;

    auto deviceOut = easyBufferCreate(hostZeros.data(), sizeof(int) * 10);
    Converter.Deserialize(cl_mem_wrapper<decltype(hostZeros)::value_type>(deviceOut, CL_MEM_READ_WRITE,
                                                                          OpenClData.context, OpenClData.queue),
                          serialIn.data(), make_options<CudaMemoryOptions::Device>());
    easyBufferRead(deviceOut, hostOut.data(), sizeof(int) * 10);
    ASSERT_EQ(hostOut, hostExpected);
}

TEST(VectorTests, DeserializeFloat4)
{
    auto Converter = midas::opencl::protobuf::converters::VectorConverter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

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

    std::vector<cl_float4> hostExpected(10);
    for (int i = 0; i < 10; i++)
        hostExpected[i] = {1.5, 2.0, 3.5, 4.0};

    std::vector<cl_float4> hostOut;
    Converter.Deserialize(hostOut, serialIn.data(),
                          make_options<CudaMemoryOptions::Device>(
                              FilledConverter(converters::Float4Converter, make_options<CudaMemoryOptions::Host>())));
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(hostOut[i].x, hostExpected[i].x);
        ASSERT_EQ(hostOut[i].y, hostExpected[i].y);
        ASSERT_EQ(hostOut[i].z, hostExpected[i].z);
        ASSERT_EQ(hostOut[i].w, hostExpected[i].w);
    }

    std::vector<cl_float4> hostZeros(10);
    for (int i = 0; i < 10; i++)
        hostZeros[i] = {0, 0, 0, 0};

    auto deviceOut = easyBufferCreate(hostZeros.data(), sizeof(cl_float4) * 10);
    Converter.Deserialize(cl_mem_wrapper<cl_float4>(deviceOut, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
                          serialIn.data(),
                          make_options<CudaMemoryOptions::Device>(
                              FilledConverter(converters::Float4Converter, make_options<CudaMemoryOptions::Host>())));
    easyBufferRead(deviceOut, hostOut.data(), sizeof(cl_float4) * 10);
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(hostOut[i].x, hostExpected[i].x);
        ASSERT_EQ(hostOut[i].y, hostExpected[i].y);
        ASSERT_EQ(hostOut[i].z, hostExpected[i].z);
        ASSERT_EQ(hostOut[i].w, hostExpected[i].w);
    }
}