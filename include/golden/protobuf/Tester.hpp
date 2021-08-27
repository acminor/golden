//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_TESTER_HPP
#define GOLDEN_PROTOBUF_TESTER_HPP

#include "golden/Tester.hpp"

#include "golden/protobuf/Comparer.hpp"
#include "golden/protobuf/Reader.hpp"
#include "golden/protobuf/ReaderZlib.hpp"
#include "golden/protobuf/Writer.hpp"
#include "golden/protobuf/WriterZlib.hpp"

namespace golden
{
    namespace protobuf
    {
        using TesterBase = golden::TesterBase<protobuf::Comparer, protobuf::Reader, protobuf::Writer>;
        using ZlibTesterBase = golden::TesterBase<protobuf::Comparer, protobuf::ReaderZlib<FILE_LOCKING_DISABLED>,
                                                  protobuf::WriterZlib<MAX_ZLIB_COMPRESSION, FILE_LOCKING_DISABLED>>;

        using Tester = golden::Tester<TesterBase>;
        using ZlibTester = golden::Tester<ZlibTesterBase>;
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_TESTER_HPP
