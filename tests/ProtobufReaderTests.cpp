//
// Created by austin on 8/2/21.
//

#include <filesystem>
#include <fstream>

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

TypedGoldenKey(FileThatDoesNotExistZlib, golden_tests::IntWrapper);
TEST(ProtobufReaderZlib, FileDoesNotExist)
{
    auto reader = golden::protobuf::Reader();
    auto value = golden_tests::IntWrapper();

    ASSERT_ANY_THROW(reader.read(FileThatDoesNotExistZlib(), value));
}

TypedGoldenKey(FileThatDoesExistZlib, golden_tests::IntWrapper);
TEST(ProtobufReaderZlib, FileExists)
{
    std::ofstream outFile(golden::GoldenUtility::PathToGolden(FileThatDoesExistZlib()));
    outFile.flush();
    outFile.close();

    auto reader = golden::protobuf::Reader();
    auto value = golden_tests::IntWrapper();

    ASSERT_NO_THROW(reader.read(FileThatDoesExistZlib(), value));
}

// TODO implement logic and test
TEST(ProtobufReader, FileExistsButIsNotReadable)
{
}