//
// Created by austin on 8/2/21.
//

#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

#include "Tester.hpp"

#include "GoldenTests.pb.h"

TypedGoldenKey(WriteThenReadNonExistingFile);
TEST(ProtobufWriter, WriteThenReadNonExistingFile)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(WriteThenReadNonExistingFile()));

    ASSERT_FALSE(
        filesystem::exists(golden::GoldenUtility::PathToGolden(WriteThenReadNonExistingFile())));

    auto writer = golden::ProtobufWriter();
    auto a = golden_tests::IntWrapper();

    a.set_value(3);
    ASSERT_NO_THROW(writer.write(WriteThenReadNonExistingFile(), a));

    auto reader = golden::ProtobufReader();
    auto b = golden_tests::IntWrapper();

    reader.read(WriteThenReadNonExistingFile(), b);
    ASSERT_EQ(a.value(), b.value());
}

TypedGoldenKey(WriteThenReadExistingFile);
TEST(ProtobufWriter, WriteThenReadExistingFile)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFile()));

    ASSERT_FALSE(
        filesystem::exists(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFile())));

    fstream outFile(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFile()));
    outFile << "Some text" << endl;
    outFile.flush();
    outFile.close();

    auto writer = golden::ProtobufWriter();
    auto a = golden_tests::IntWrapper();

    a.set_value(3);
    ASSERT_NO_THROW(writer.write(WriteThenReadExistingFile(), a));

    auto reader = golden::ProtobufReader();
    auto b = golden_tests::IntWrapper();

    reader.read(WriteThenReadExistingFile(), b);
    ASSERT_EQ(a.value(), b.value());
}