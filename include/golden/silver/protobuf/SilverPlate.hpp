//
// Created by austin on 8/3/21.
//

#ifndef GOLDEN_SILVER_PROTOBUF_SILVERPLATE_HPP
#define GOLDEN_SILVER_PROTOBUF_SILVERPLATE_HPP

#include "golden/protobuf/Reader.hpp"
#include "golden/protobuf/ReaderZlib.hpp"
#include "golden/protobuf/Writer.hpp"
#include "golden/protobuf/WriterZlib.hpp"
#include "golden/silver/SilverPlate.hpp"

namespace golden
{
    namespace silver
    {
        namespace protobuf
        {
            using SilverPlateBase = silver::SilverPlateBase<golden::protobuf::Reader, golden::protobuf::Writer>;
            using SilverPlateBaseZlib =
                silver::SilverPlateBase<golden::protobuf::ReaderZlib<golden::protobuf::FILE_LOCKING_DISABLED>,
                                        golden::protobuf::WriterZlib<golden::protobuf::MAX_ZLIB_COMPRESSION,
                                                                     golden::protobuf::FILE_LOCKING_DISABLED>>;

            using SilverPlate = silver::SilverPlate<SilverPlateBase>;
            using SilverPlateZlib = silver::SilverPlate<SilverPlateBaseZlib>;
        } // namespace protobuf
    }     // namespace silver
} // namespace golden

#endif // GOLDEN_SILVER_PROTOBUF_SILVERPLATE_HPP
