//
// Created by austin on 8/2/21.
//

#include <cfloat>

#include "gtest/gtest.h"

#include "Tester.hpp"
#include "GoldenTests.pb.h"

enum Comparison {
    LESS_THAN = -1,
    EQUAL = 0,
    GREATER_THAN = 1,
};

/**
 * Uninitialized data compares equal with each other.
 */
TEST(ProtobufComparer, TwoUnsetMessages)
{
    auto a = golden_tests::IntPairWrapper();
    auto b = golden_tests::IntPairWrapper();

    auto comparer = golden::ProtobufComparer();

    ASSERT_EQ(comparer.compare(a, b), EQUAL);
}

/**
 * Uninitialized and initialized data does not compare with each other.
 * - unless initial data is default for uninitalized data (e.g. 0)
 */
TEST(ProtobufComparer, OneSetOneUnsetMessage)
{
    auto a = golden_tests::IntWrapper();
    auto b = golden_tests::IntWrapper();
    auto comparer = golden::ProtobufComparer();

    b.set_value(1);
    ASSERT_NE(comparer.compare(a, b), EQUAL);

    b.set_value(0);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);
}

TEST(ProtobufComparer, EqualOneFieldMessage)
{
    auto a = golden_tests::IntWrapper();
    auto b = golden_tests::IntWrapper();
    auto comparer = golden::ProtobufComparer();

    a.set_value(0);
    b.set_value(0);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);

    a.set_value(-1);
    b.set_value(-1);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);

    a.set_value(1);
    b.set_value(1);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);
}

TEST(ProtobufComparer, NonEqualOneFieldMessage)
{
    auto a = golden_tests::IntWrapper();
    auto b = golden_tests::IntWrapper();
    auto comparer = golden::ProtobufComparer();

    a.set_value(0);
    b.set_value(-1);
    ASSERT_NE(comparer.compare(a, b), EQUAL);

    a.set_value(1);
    b.set_value(0);
    ASSERT_NE(comparer.compare(a, b), EQUAL);

    a.set_value(1);
    b.set_value(-1);
    ASSERT_NE(comparer.compare(a, b), EQUAL);
}

TEST(ProtobufComparer, FloatMessageExact)
{
    auto a = golden_tests::FloatWrapper();
    auto b = golden_tests::FloatWrapper();
    auto comparer = golden::ProtobufComparer();

    a.set_value(0.0);
    b.set_value(0.0);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);

    a.set_value(0.5);
    b.set_value(0.5);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);

    a.set_value(-0.5);
    b.set_value(-0.5);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);
}

TEST(ProtobufComparer, FloatMessageNotEqual)
{
    auto a = golden_tests::FloatWrapper();
    auto b = golden_tests::FloatWrapper();
    auto comparer = golden::ProtobufComparer();

    a.set_value(0.5);
    b.set_value(0.0);
    ASSERT_NE(comparer.compare(a, b), EQUAL);

    a.set_value(0.0);
    b.set_value(-0.5);
    ASSERT_NE(comparer.compare(a, b), EQUAL);

    a.set_value(0.5);
    b.set_value(-0.5);
    ASSERT_NE(comparer.compare(a, b), EQUAL);
}

/**
 * See Google's Protobuf Almost Equals implementation
 * which we use in the FieldComparator of the MessageDifferencer.
 * - https://github.com/protocolbuffers/protobuf/blob/79370f1ffa81a8394c41a99a32ab1452f923242e/src/google/protobuf/stubs/mathutil.h#L69
 */
TEST(ProtobufComparer, FloatMessageApproximate)
{
    auto a = golden_tests::FloatWrapper();
    auto b = golden_tests::FloatWrapper();
    auto comparer = golden::ProtobufComparer();

    a.set_value(3.0f);
    b.set_value(3.0f - 30*FLT_EPSILON);
    ASSERT_EQ(comparer.compare(a, b), EQUAL);

    a.set_value(3.0f);
    b.set_value(3.0f - 31*FLT_EPSILON);
    ASSERT_NE(comparer.compare(a, b), EQUAL);
}