//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <midas/opencl/Dim3Converter.hpp>
#include <midas/opencl/OpenCLBase.hpp>
#include <midas/opencl/OpenCLHelpers.hpp>

#include <MidasTests.pb.h>
#include <opencl/TestsUtility.hpp>

TEST(Dim3Tests, Serialize)
{
    auto Converter = midas::opencl::protobuf::converters::Dim3Converter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    google::protobuf::util::MessageDifferencer messageDifferencer;
    midas_tests::Dim3Wrapper serialExpected;
    {
        auto data = serialExpected.mutable_data();
        data->set_x(1);
        data->set_y(2);
        data->set_z(3);
    }

    size_t hostIn[] = {1, 2, 3};

    midas_tests::Dim3Wrapper serialOut;
    Converter.Serialize(dim3(hostIn), serialOut.mutable_data(), make_options<CudaMemoryOptions::Host>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut));
    }

    auto deviceIn = easyBufferCreate(&hostIn, sizeof(hostIn));
    Converter.Serialize(cl_mem_wrapper<size_t>(deviceIn, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
                        serialOut.mutable_data(), make_options<CudaMemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

TEST(Dim3Tests, Deserialize)
{
    auto Converter = midas::opencl::protobuf::converters::Dim3Converter;

    using namespace midas::opencl::protobuf;
    using namespace midas::opencl;

    midas_tests::Dim3Wrapper serialIn;
    {
        auto data = serialIn.mutable_data();
        data->set_x(1);
        data->set_y(2);
        data->set_z(3);
    }

    size_t hostExpected[] = {1, 2, 3};

    size_t hostOut[3];
    Converter.Deserialize(dim3Ref(hostOut), serialIn.data(), make_options<CudaMemoryOptions::Host>());

    for (int i = 0; i < 3; i++)
        ASSERT_EQ(hostOut[i], hostExpected[i]);

    size_t hostZeros[] = {0, 0, 0};
    auto deviceOut = easyBufferCreate(&hostZeros, sizeof(hostZeros));
    Converter.Deserialize(cl_mem_wrapper<size_t>(deviceOut, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
                          serialIn.data(), make_options<CudaMemoryOptions::Device>());

    easyBufferRead(deviceOut, &hostOut, sizeof(hostOut));

    for (int i = 0; i < 3; i++)
        ASSERT_EQ(hostOut[i], hostExpected[i]);
}