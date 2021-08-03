//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_READER_HPP
#define GOLDEN_PROTOBUF_READER_HPP

#include "golden/Utility.hpp"

namespace golden
{
    namespace protobuf
    {
        class Reader
        {
          public:
            template <typename GoldenKey> inline int read(GoldenKey key, typename GoldenKey::MessageType &object)
            {
                auto in = GoldenUtility::ReadFromGolden(key);

                object.ParseFromIstream(&in);
                in.close();

                return 0;
            }
        };
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_READER_HPP
