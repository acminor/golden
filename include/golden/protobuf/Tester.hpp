//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_TESTER_HPP
#define GOLDEN_PROTOBUF_TESTER_HPP

#include "golden/Tester.hpp"

#include "golden/protobuf/Comparer.hpp"
#include "golden/protobuf/Reader.hpp"
#include "golden/protobuf/Writer.hpp"

namespace golden
{
    namespace protobuf
    {
        using TesterBase = golden::TesterBase<protobuf::Comparer, protobuf::Reader, protobuf::Writer>;
        using Tester = golden::Tester<TesterBase>;
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_TESTER_HPP
