//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_READER_ZLIB_HPP
#define GOLDEN_PROTOBUF_READER_ZLIB_HPP

#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>

#include "golden/Utility.hpp"
#include "golden/protobuf/Zlib.h"

namespace golden
{
    namespace protobuf
    {
        template <bool isFileLockingEnabled>
        class ReaderZlib
        {
          public:
            ReaderZlib() : m_isFileLockingEnabled(isFileLockingEnabled)
            {
            }
            /**
             * See WriterZlib for a description of the locking going on here.
             */
            template <typename GoldenKey>
            inline int read(GoldenKey key, typename GoldenKey::MessageType &object)
            {
                auto in = GoldenUtility::ReadFromGolden(key);

                auto fd = open(GoldenUtility::PathToGolden(key).c_str(), O_RDONLY);
                if (isFileLockingEnabled)
                    flock(fd, LOCK_EX);
                google::protobuf::io::FileInputStream fileInputStream(fd);
                google::protobuf::io::GzipInputStream gzipInputStream(&fileInputStream);

                object.ParseFromZeroCopyStream(&gzipInputStream);

                if (isFileLockingEnabled)
                    flock(fd, LOCK_UN);
                fileInputStream.Close();

                return 0;
            }

          private:
            const bool m_isFileLockingEnabled;
        };
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_READER_ZLIB_HPP
