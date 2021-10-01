//
// Created by acminor on 9/23/21.
//

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <midas/cuda/MallocInterop.hpp>
#include <midas/cuda/MallocInteropArrayConverter.hpp>

#include <vector>

TEST(MallocInterop, GeneralTests)
{
    int *len10Data;
    cudaMalloc((void **)&len10Data, 10 * sizeof(int));

    int *len20Data;
    cudaMalloc((void **)&len20Data, 20 * sizeof(int));

    int *len1Data;
    cudaMalloc((void **)&len1Data, sizeof(int));

    ASSERT_EQ(cudaGetArraySize(len10Data), 10 * sizeof(int));
    ASSERT_EQ(cudaGetArraySize(len20Data), 20 * sizeof(int));
    ASSERT_EQ(cudaGetArraySize(len1Data), 1 * sizeof(int));
}

TEST(MallocInterop, Serialize)
{
    auto Converter = midas::cuda::protobuf::converters::MallocInteropArrayConverter;

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
    int *deviceIn;
    cudaMalloc((void **)&deviceIn, 10 * sizeof(hostIn[0]));
    cudaMemcpy(deviceIn, hostIn.data(), 10 * sizeof(hostIn[0]), cudaMemcpyHostToDevice);
    Converter.Serialize(
        deviceIn, [&](const auto &x) { serialOut.add_data(x); }, make_options<MemoryOptions::Device>());

    {
        std::string result;
        messageDifferencer.ReportDifferencesToString(&result);
        ASSERT_TRUE(messageDifferencer.Compare(serialExpected, serialOut)) << result << std::endl;
    }
}

// TODO more advanced tests to make sure memory is okay (flags, values, etc.)
