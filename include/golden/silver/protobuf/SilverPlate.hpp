//
// Created by austin on 8/3/21.
//

#ifndef GOLDEN_SILVER_PROTOBUF_SILVERPLATE_HPP
#define GOLDEN_SILVER_PROTOBUF_SILVERPLATE_HPP

#include "golden/protobuf/Reader.hpp"
#include "golden/protobuf/Writer.hpp"
#include "golden/silver/SilverPlate.hpp"

namespace golden
{
    namespace silver
    {
        namespace protobuf
        {
            using SilverPlate = SilverPlate<SilverPlateBase<golden::protobuf::Reader, golden::protobuf::Writer>>;
        }
    } // namespace silver
} // namespace golden

#endif // GOLDEN_SILVER_PROTOBUF_SILVERPLATE_HPP
