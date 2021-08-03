//
// Created by austin on 8/2/21.
//

#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"

#include "Golden.hpp"

#include "GoldenTests.pb.h"

TypedGoldenKey(FileThatDoesNotExist, golden_tests::IntWrapper);
TEST(ProtobufReader, FileDoesNotExist)
{
    auto reader = golden::protobuf::Reader();
    auto value = golden_tests::IntWrapper();

    ASSERT_ANY_THROW(reader.read(FileThatDoesNotExist(), value));
}

TypedGoldenKey(FileThatDoesExist, golden_tests::IntWrapper);
TEST(ProtobufReader, FileExists)
{
    std::ofstream outFile(golden::GoldenUtility::PathToGolden(FileThatDoesExist()));
    outFile.flush();
    outFile.close();

    auto reader = golden::protobuf::Reader();
    auto value = golden_tests::IntWrapper();

    ASSERT_NO_THROW(reader.read(FileThatDoesExist(), value));
}

// TODO implement logic and test
TEST(ProtobufReader, FileExistsButIsNotReadable)
{
}