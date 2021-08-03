//
// Created by austin on 8/3/21.
//

#include <filesystem>

#include "gtest/gtest.h"

#include "Golden.hpp"
#include "GoldenTests.pb.h"

TypedGoldenKey(ConstructSnapshot, golden_tests::IntPairWrapper);
TEST(ProtobufSilver, Construct)
{
    auto silverer = golden::silver::ProtobufSilverPlate();

    auto message = golden_tests::IntPairWrapper();
    message.set_value(1);
    message.set_value2(2);

    auto recoveryFunction = [](auto message) {};

    auto messageFunction = []() { return golden_tests::IntPairWrapper(); };

    silverer.Silver(ConstructSnapshot(), message);
    silverer.Silver(ConstructSnapshot(), messageFunction);
    silverer.Desilver(ConstructSnapshot(), recoveryFunction);
}

TypedGoldenKey(SilverSnapshot1, golden_tests::IntPairWrapper);
TypedGoldenKey(SilverSnapshot2, golden_tests::IntPairWrapper);
TEST(ProtobufSilver, Snapshot)
{
    namespace fs = std::filesystem;
    auto message = golden_tests::IntPairWrapper();
    message.set_value(1);
    message.set_value2(2);

    auto a = 3;
    auto b = 4;
    auto messageFunction = [&a, &b]() {
        auto message = golden_tests::IntPairWrapper();
        message.set_value(a);
        message.set_value2(b);

        return message;
    };

    auto protobufReader = golden::protobuf::Reader();
    decltype(message) result;

    auto silverer = golden::silver::ProtobufSilverPlate();

    fs::remove(golden::GoldenUtility::PathToGolden(SilverSnapshot1()));
    silverer.Silver(SilverSnapshot1(), message);
    ASSERT_TRUE(fs::exists(golden::GoldenUtility::PathToGolden(SilverSnapshot1())));
    protobufReader.read(SilverSnapshot1(), result);
    ASSERT_EQ(1, result.value());
    ASSERT_EQ(2, result.value2());

    fs::remove(golden::GoldenUtility::PathToGolden(SilverSnapshot2()));
    silverer.Silver(SilverSnapshot2(), messageFunction);
    ASSERT_TRUE(fs::exists(golden::GoldenUtility::PathToGolden(SilverSnapshot2())));
    protobufReader.read(SilverSnapshot2(), result);
    ASSERT_EQ(3, result.value());
    ASSERT_EQ(4, result.value2());
}

TypedGoldenKey(RecoverSnapshotV2, golden_tests::IntPairWrapper);
TEST(ProtobufSilver, Recovery)
{
    namespace fs = std::filesystem;

    auto a = 3;
    auto b = 4;
    auto messageFunction = [&a, &b]() {
        auto message = golden_tests::IntPairWrapper();
        message.set_value(a);
        message.set_value2(b);

        return message;
    };

    auto silverer = golden::silver::ProtobufSilverPlate();

    fs::remove(golden::GoldenUtility::PathToGolden(RecoverSnapshotV2()));
    silverer.Silver(RecoverSnapshotV2(), messageFunction);
    ASSERT_TRUE(fs::exists(golden::GoldenUtility::PathToGolden(RecoverSnapshotV2())));

    auto c = -1;
    auto d = -1;

    auto recoveryFunction = [&](const golden_tests::IntPairWrapper &a) {
        c = a.value();
        d = a.value2();
    };

    silverer.Desilver(RecoverSnapshotV2(), recoveryFunction);
    ASSERT_EQ(3, c);
    ASSERT_EQ(4, d);
}