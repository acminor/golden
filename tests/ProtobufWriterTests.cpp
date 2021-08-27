//
// Created by austin on 8/2/21.
//

#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

#include "Golden.hpp"

#include "GoldenTests.pb.h"

TypedGoldenKey(WriteThenReadNonExistingFile, golden_tests::IntWrapper);
TEST(ProtobufWriter, WriteThenReadNonExistingFile)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(WriteThenReadNonExistingFile()));

    ASSERT_FALSE(filesystem::exists(golden::GoldenUtility::PathToGolden(WriteThenReadNonExistingFile())));

    auto writer = golden::protobuf::Writer();
    auto a = golden_tests::IntWrapper();

    a.set_value(3);
    ASSERT_NO_THROW(writer.write(WriteThenReadNonExistingFile(), a));

    auto reader = golden::protobuf::Reader();
    auto b = golden_tests::IntWrapper();

    reader.read(WriteThenReadNonExistingFile(), b);
    ASSERT_EQ(a.value(), b.value());
}

TypedGoldenKey(WriteThenReadExistingFile, golden_tests::IntWrapper);
TEST(ProtobufWriter, WriteThenReadExistingFile)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFile()));

    ASSERT_FALSE(filesystem::exists(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFile())));

    fstream outFile(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFile()));
    outFile << "Some text" << endl;
    outFile.flush();
    outFile.close();

    auto writer = golden::protobuf::Writer();
    auto a = golden_tests::IntWrapper();

    a.set_value(3);
    ASSERT_NO_THROW(writer.write(WriteThenReadExistingFile(), a));

    auto reader = golden::protobuf::Reader();
    auto b = golden_tests::IntWrapper();

    reader.read(WriteThenReadExistingFile(), b);
    ASSERT_EQ(a.value(), b.value());
}

TypedGoldenKey(WriteThenReadNonExistingFileZlib, golden_tests::IntWrapper);
TEST(ProtobufWriterZlib, WriteThenReadNonExistingFile)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(WriteThenReadNonExistingFileZlib()));

    ASSERT_FALSE(filesystem::exists(golden::GoldenUtility::PathToGolden(WriteThenReadNonExistingFileZlib())));

    auto writer =
        golden::protobuf::WriterZlib<golden::protobuf::MAX_ZLIB_COMPRESSION, golden::protobuf::FILE_LOCKING_DISABLED>();
    auto a = golden_tests::IntWrapper();

    a.set_value(3);
    ASSERT_NO_THROW(writer.write(WriteThenReadNonExistingFileZlib(), a));

    auto reader = golden::protobuf::ReaderZlib<golden::protobuf::FILE_LOCKING_DISABLED>();
    auto b = golden_tests::IntWrapper();

    reader.read(WriteThenReadNonExistingFileZlib(), b);
    ASSERT_EQ(a.value(), b.value());
}

TypedGoldenKey(WriteThenReadExistingFileZlib, golden_tests::IntWrapper);
TEST(ProtobufWriterZlib, WriteThenReadExistingFile)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFileZlib()));

    ASSERT_FALSE(filesystem::exists(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFileZlib())));

    fstream outFile(golden::GoldenUtility::PathToGolden(WriteThenReadExistingFileZlib()));
    outFile << "Some text" << endl;
    outFile.flush();
    outFile.close();

    auto writer =
        golden::protobuf::WriterZlib<golden::protobuf::MAX_ZLIB_COMPRESSION, golden::protobuf::FILE_LOCKING_DISABLED>();
    auto a = golden_tests::IntWrapper();

    a.set_value(3);
    ASSERT_NO_THROW(writer.write(WriteThenReadExistingFileZlib(), a));

    auto reader = golden::protobuf::ReaderZlib<golden::protobuf::FILE_LOCKING_DISABLED>();
    auto b = golden_tests::IntWrapper();

    reader.read(WriteThenReadExistingFileZlib(), b);
    ASSERT_EQ(a.value(), b.value());
}

TypedGoldenKey(CompressionTestZlib, golden_tests::IntArray);
TypedGoldenKey(CompressionTestZlibUncompressed, golden_tests::IntArray);
TEST(ProtobufWriterZlib, CompressionTest)
{
    using namespace std;

    filesystem::remove(golden::GoldenUtility::PathToGolden(CompressionTestZlibUncompressed()));
    ASSERT_FALSE(filesystem::exists(golden::GoldenUtility::PathToGolden(CompressionTestZlibUncompressed())));

    filesystem::remove(golden::GoldenUtility::PathToGolden(CompressionTestZlib()));
    ASSERT_FALSE(filesystem::exists(golden::GoldenUtility::PathToGolden(CompressionTestZlib())));

    auto writer = golden::protobuf::Writer();
    auto writerZlib =
        golden::protobuf::WriterZlib<golden::protobuf::MAX_ZLIB_COMPRESSION, golden::protobuf::FILE_LOCKING_DISABLED>();
    auto a = golden_tests::IntArray();

    for (int i = 0; i < 10000; i++)
        a.add_data(0);

    ASSERT_NO_THROW(writer.write(CompressionTestZlibUncompressed(), a));
    ASSERT_NO_THROW(writerZlib.write(CompressionTestZlib(), a));

    auto uncompressedSize =
        filesystem::file_size(golden::GoldenUtility::PathToGolden(CompressionTestZlibUncompressed()));
    auto compressedSize = filesystem::file_size(golden::GoldenUtility::PathToGolden(CompressionTestZlib()));

    // At least 1000 bytes smaller for compressed version
    ASSERT_GE(uncompressedSize - compressedSize, 1000);
}

TypedGoldenKey(MultiFileAccessZlib, golden_tests::IntWrapper);
TEST(ProtobufWriterZlib, MultiFileAccess)
{
    // TODO C++ multithreaded test
}