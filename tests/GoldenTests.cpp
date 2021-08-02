//
// Created by austin on 7/30/21.
//

#include <filesystem>

#include "gtest/gtest.h"

#include "Tester.hpp"

#include "GoldenTests.pb.h"

TypedGoldenKey(IntegrationTest);
TEST(GoldenTests, IntegrationTest)
{
    namespace fs = std::filesystem;

    auto tester = golden::ProtobufTester();

    auto a = golden_tests::IntWrapper();
    auto b = golden_tests::IntWrapper();
    a.set_value(3);
    b.set_value(2);

    auto goldenPath = golden::GoldenUtility::PathToGolden(IntegrationTest());
    fs::remove(goldenPath);
    ASSERT_FALSE(fs::exists(goldenPath));

    auto result = tester.validate(IntegrationTest(), a);
    ASSERT_TRUE(result.isSavedGolden());

    result = tester.validate(IntegrationTest(), a);
    ASSERT_TRUE(result.isSuccess());

    result = tester.validate(IntegrationTest(), b);
    ASSERT_TRUE(result.isFailure());

    ASSERT_NO_THROW(tester.require(IntegrationTest(), a));
    ASSERT_ANY_THROW(tester.require(IntegrationTest(), b));
}

// TODO Unit text Utility functions and classes

TEST(GoldenTests, GoldenUtility)
{
}

TEST(GoldenTests, GoldenKeyUtility)
{
}

TEST(GoldenTests, GoldenKey)
{
}

TEST(GoldenTests, GoldenResult)
{
}