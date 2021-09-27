//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <midas/opencl/MallocInterop.hpp>
#include <midas/opencl/MallocInteropArrayConverter.hpp>

#include <opencl/TestsUtility.hpp>

#include <vector>

TEST(MallocInterop, GeneralTests)
{
    std::vector<int> exampleData(100);
    auto len10Data = easyBufferCreate(exampleData.data(), 10);

    auto len20Data = easyBufferCreate(exampleData.data(), 20);

    auto len1Data = easyBufferCreate(exampleData.data(), 1);

    ASSERT_EQ(openclGetArraySize(len10Data), 10);
    ASSERT_EQ(openclGetArraySize(len20Data), 20);
    ASSERT_EQ(openclGetArraySize(len1Data), 1);
}

TEST(MallocInterop, Serialize)
{
    InitializeOpenCL();

    auto Converter = midas::opencl::protobuf::converters::MallocInteropArrayConverter;

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
    auto deviceIn = easyBufferCreate(hostIn.data(), sizeof(int) * 10);
    Converter.Serialize(
        cl_mem_wrapper<int>(deviceIn, CL_MEM_READ_WRITE, OpenClData.context, OpenClData.queue),
        [&](const auto &x) { serialOut.add_data(x); }, make_options<CudaMemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

// TODO more advanced tests to make sure memory is okay (flags, values, etc.)
