//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_WRITER_ZLIB_HPP
#define GOLDEN_PROTOBUF_WRITER_ZLIB_HPP

#include <string>
#include <type_traits>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>

#include "golden/Utility.hpp"
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>

#include "golden/protobuf/Zlib.h"

namespace golden
{
    namespace protobuf
    {
        template <typename Message>
        using IsMessageType = std::enable_if_t<std::is_base_of_v<google::protobuf::Message, Message>, bool>;

        template <int zlibCompressionLevel, bool isFileLockingEnabled> class WriterZlib
        {
          public:
            WriterZlib() : m_isFileLockingEnabled(isFileLockingEnabled), m_compressionLevel(zlibCompressionLevel)
            {
            }
            /**
             * Note: current file locking may not work if process is forked and parent process
             *       also writes to the same file a child is writing to. See `man flock(2)`.
             */
            template <typename GoldenKey, IsMessageType<typename GoldenKey::MessageType> = true>
            inline int write(GoldenKey key, const typename GoldenKey::MessageType object)
            {
                // TODO needs unittest for this
                GoldenUtility::InitGoldenDirectory();

                auto fd = open(GoldenUtility::PathToGolden(key).c_str(), O_CREAT | O_TRUNC | O_WRONLY);
                fchmod(fd, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
                if (isFileLockingEnabled)
                    flock(fd, LOCK_EX); // prevent double writing file in multi-threaded context
                google::protobuf::io::FileOutputStream fileOutputStream(fd);

                auto gzipOptions = google::protobuf::io::GzipOutputStream::Options();
                gzipOptions.compression_level = zlibCompressionLevel;
                gzipOptions.format = google::protobuf::io::GzipOutputStream::ZLIB;
                google::protobuf::io::GzipOutputStream gzipOutputStream(&fileOutputStream, gzipOptions);

                object.SerializeToZeroCopyStream(&gzipOutputStream);

                // flush then unlock, then close
                // - this is required because we want all current data to exist before
                //   rewriting and fileOutputStream.Close closes the underlying file
                gzipOutputStream.Flush();
                fileOutputStream.Flush();

                if (isFileLockingEnabled)
                    flock(fd, LOCK_UN);

                gzipOutputStream.Close();
                fileOutputStream.Close();

                return 0;
            }

            const bool m_isFileLockingEnabled;
            const unsigned char m_compressionLevel;
        };
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_WRITER_ZLIB_HPP
