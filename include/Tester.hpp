//
// Created by austin on 7/30/21.
//

#ifndef GOLDEN_TESTER_HPP
#define GOLDEN_TESTER_HPP

#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

#include "Config.hpp"

#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/message_differencer.h"

namespace golden
{
    // NOTE not used yet
    namespace protobuf
    {
        template <typename T> std::string WriteObject(T object);

        template <> inline std::string WriteObject(int o)
        {
            return "";
        }
    } // namespace protobuf

    struct GoldenResult
    {
      public:
        inline void setSuccess()
        {
            m_type = ResultType::Success;
            m_data = 0;
        }

        inline bool isSuccess()
        {
            return m_type == ResultType::Success;
        }

        inline void setFailure(int errorCode)
        {
            m_type = ResultType::Failure;
            m_data = errorCode;
        }

        inline bool isFailure()
        {
            return m_type == ResultType::Failure;
        }

        // TODO implment this functionality
        inline void setExecption(const std::string &message)
        {
            m_type = ResultType::Exception;
            m_data = message;
        }

        inline void setSavedGolden()
        {
            m_type = ResultType::SavedGolden;
            m_data = 0;
        }

        inline bool isSavedGolden()
        {
            return m_type == ResultType::SavedGolden;
        }

      private:
        enum class ResultType
        {
            Success = 0,
            Failure = 1,
            Exception = 2,
            SavedGolden = 3,
        };

        ResultType m_type;
        std::variant<int, std::string> m_data;
    };

    class GoldenKeyUtility
    {
      public:
        inline static std::string getPath(const std::string &path)
        {
            return path;
        }
    };

#define TypedGoldenKey(NAME)                                                                                           \
    class NAME                                                                                                         \
    {                                                                                                                  \
      public:                                                                                                          \
        inline std::string getPath()                                                                                          \
        {                                                                                                              \
            using namespace golden;                                                                                    \
            return GoldenKeyUtility::getPath(#NAME);                                                                   \
        }                                                                                                              \
    }

    TypedGoldenKey(TestKey);

    using namespace google::protobuf::util;
    class ProtobufComparer
    {
      public:
        inline ProtobufComparer() : m_messageDifferencer(), m_fieldComparator()
        {
            m_fieldComparator.set_float_comparison(DefaultFieldComparator::APPROXIMATE);
            m_messageDifferencer.set_field_comparator(&m_fieldComparator);
            m_messageDifferencer.set_message_field_comparison(MessageDifferencer::EQUAL);
            m_messageDifferencer.set_scope(MessageDifferencer::FULL);
        }

        template <typename T> inline int compare(T a, T b)
        {
            auto equal = m_messageDifferencer.Compare(a, b);

            if (equal)
                return 0;
            else
                return -1;
        }

      private:
        DefaultFieldComparator m_fieldComparator;
        MessageDifferencer m_messageDifferencer;
    };

    class GoldenUtility
    {
      public:
        template <typename GoldenKey> inline static void WriteToGolden(GoldenKey key, const std::string &buffer)
        {
            if (!is_directory(GOLDEN_PATH))
                create_directory(GOLDEN_PATH);

            std::ofstream out(PathToGolden(key));
            out << buffer;
            out.flush();
            out.close();
        }

        template <typename GoldenKey> inline static std::ifstream ReadFromGolden(GoldenKey key)
        {
            if (!std::filesystem::exists(PathToGolden(key)))
                throw "TODO implement proper exception";

            std::ifstream in(PathToGolden(key), std::ios_base::in | std::ios_base::binary);
            return in;
        }

        template <typename GoldenKey> inline static bool GoldenExists(GoldenKey key)
        {
            return std::filesystem::exists(PathToGolden(key));
        }

        template <typename GoldenKey> inline static std::filesystem::path PathToGolden(GoldenKey key)
        {
            return GoldenUtility::GOLDEN_PATH / key.getPath();
        }

      private:
        inline static const std::filesystem::path GOLDEN_PATH = std::filesystem::path(golden_GOLDEN_STORAGE_PATH);
    };

    class ProtobufReader
    {
      public:
        template <typename T, typename GoldenKey> inline int read(GoldenKey key, T &object)
        {
            auto in = GoldenUtility::ReadFromGolden(key);

            object.ParseFromIstream(&in);
            in.close();

            return 0;
        }
    };

    class ProtobufWriter
    {
      public:
        template <typename T, typename GoldenKey> inline int write(GoldenKey key, T object)
        {
            // std::string buffer = golden::protobuf::WriteObject(object);
            std::string buffer;
            object.SerializeToString(&buffer);

            GoldenUtility::WriteToGolden(key, buffer);

            return 0;
        }
    };

    template <typename Comparer = ProtobufComparer, typename Reader = ProtobufReader, typename Writer = ProtobufWriter>
    class ProtobufTester
    {
      public:
        inline ProtobufTester() : m_writer(), m_comparer(), m_reader()
        {
        }

        /**
         * Validates that two fields are equal
         * @tparam T
         * @param key
         * @param result
         * @return
         */
        template <typename GoldenKey, typename T> inline GoldenResult validate(GoldenKey key, const T result)
        {
            return _validate(key, result);
        }

        /**
         * Require operates like validate but throws an exception.
         * @tparam T
         * @param key
         * @param result
         * @return
         */
        template <typename GoldenKey, typename T> inline void require(GoldenKey key, const T result)
        {
            auto compareResult = _validate(key, result);

            if (!compareResult.isSuccess())
                throw "TODO Implement Require Execption";
        }

      private:
        template <typename GoldenKey, typename T> inline GoldenResult _validate(GoldenKey key, const T result)
        {
            auto output = GoldenResult();
            if (GoldenUtility::GoldenExists(key))
            {
                T golden;
                m_reader.read(key, golden);

                auto compareResult = m_comparer.compare(golden, result);
                if (compareResult == 0)
                    output.setSuccess();
                else
                    output.setFailure(compareResult);
            }
            else
            {
                m_writer.write(key, result);
                output.setSavedGolden();
            }

            return output;
        }

        Comparer m_comparer;
        Reader m_reader;
        Writer m_writer;
    };
} // namespace golden

#endif // GOLDEN_TESTER_HPP
