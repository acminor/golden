//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_WRITER_HPP
#define GOLDEN_PROTOBUF_WRITER_HPP

#include <string>

#include "golden/Utility.hpp"

namespace golden
{
    namespace protobuf
    {
        class Writer
        {
          public:
            template <typename T, typename GoldenKey> inline int write(GoldenKey key, T object)
            {
                std::string buffer;
                object.SerializeToString(&buffer);

                GoldenUtility::WriteToGolden(key, buffer);

                return 0;
            }
        };
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_WRITER_HPP
