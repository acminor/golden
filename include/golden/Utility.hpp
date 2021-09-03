//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_UTILITY_HPP
#define GOLDEN_UTILITY_HPP

#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

#include "golden/Config.hpp"

namespace golden
{
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

    template <const char *Path, typename MsgType> class GoldenKeyType
    {
      public:
        using MessageType = MsgType;
        static inline std::string getPath()
        {
            using namespace golden;
            return GoldenKeyUtility::getPath(Path);
        }
    };

#define TypedGoldenKeyEmptyArg

#define TypedGoldenKeyNameName(NAME, PREFIX, POSTFIX) PREFIX##NAME##POSTFIX

#define TypedGoldenKeyName(NAME, PREFIX, POSTFIX) const char TypedGoldenKeyNameName(NAME, PREFIX, POSTFIX)[] = #NAME

#define TypedGoldenKeyType(NAME, PREFIX, POSTFIX, MESSAGE_TYPE)                                                        \
    using NAME = golden::GoldenKeyType<TypedGoldenKeyNameName(NAME, PREFIX, POSTFIX), MESSAGE_TYPE>

#define TypedGoldenKeyWithPrefixPostfix(NAME, PREFIX, POSTFIX, MESSAGE_TYPE)                                           \
    TypedGoldenKeyName(NAME, PREFIX, POSTFIX);                                                                         \
    TypedGoldenKeyType(NAME, PREFIX, POSTFIX, MESSAGE_TYPE);

#define TypedGoldenKey(NAME, MESSAGE_TYPE)                                                                             \
    TypedGoldenKeyWithPrefixPostfix(NAME, TypedGoldenKeyEmptyArg, Name, MESSAGE_TYPE)

    // TODO needs unittest
    template <typename GoldenKeyIn> class GoldenFailureKeyTransformer
    {
      public:
        using GoldenKey = GoldenKeyIn;
        using MessageType = typename GoldenKey::MessageType;
        GoldenFailureKeyTransformer(const GoldenKey &key) : m_path(key.getPath())
        {
        }

        inline std::string getPath()
        {
            using namespace golden;
            return GoldenKeyUtility::getPath(m_path) + "_failed";
        }

      private:
        std::string m_path;
    };

    class GoldenUtility
    {
      public:
        template <typename GoldenKey> inline static void WriteToGolden(GoldenKey key, const std::string &buffer)
        {
            if (!is_directory(GOLDEN_PATH))
                create_directory(GOLDEN_PATH);

            std::ofstream out(PathToGolden(key), std::ofstream::binary);
            out << buffer;
            out.flush();
            out.close();
        }

        template <typename GoldenKey> inline static std::ifstream ReadFromGolden(GoldenKey key)
        {
            if (!std::filesystem::exists(PathToGolden(key)))
                throw "TODO implement proper exception";

            std::ifstream in(PathToGolden(key), std::ios_base::binary);
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
} // namespace golden

#endif // GOLDEN_UTILITY_HPP
