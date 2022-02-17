//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_UTILITY_HPP
#define GOLDEN_UTILITY_HPP

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <variant>

#include "golden/Config.hpp"

namespace fs = std::filesystem;
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

    template <const char *Path, typename MsgType>
    class GoldenKeyType
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
    template <typename GoldenKeyIn>
    class GoldenFailureKeyTransformer
    {
      public:
        using GoldenKey = GoldenKeyIn;
        using MessageType = typename GoldenKey::MessageType;
        explicit GoldenFailureKeyTransformer(const GoldenKey &key) : m_path(key.getPath())
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

    class GoldenException : public std::exception
    {
      public:
        explicit GoldenException(std::string message) : m_message(message)
        {
        }

        const char *what() const noexcept override
        {
            return m_message.c_str();
        }

      private:
        std::string m_message;
    };

    class GoldenUtility
    {
      public:
        template <typename GoldenKey>
        inline static void WriteToGolden(GoldenKey key, const std::string &buffer)
        {
            std::ofstream out(PathToGolden(key), std::ofstream::binary);
            out << buffer;
            out.flush();
            out.close();
        }

        template <typename GoldenKey>
        inline static std::ifstream ReadFromGolden(GoldenKey key)
        {
            if (!fs::exists(PathToGolden(key)))
                throw GoldenException(PathToGolden(key).string());

            std::ifstream in(PathToGolden(key), std::ios_base::binary);
            return in;
        }

        template <typename GoldenKey>
        inline static bool GoldenExists(GoldenKey key)
        {
            return fs::exists(PathToGolden(key));
        }

        template <typename GoldenKey>
        inline static fs::path PathToGolden(GoldenKey key)
        {
            return GoldenUtility::GOLDEN_PATH / key.getPath();
        }

      private:
        inline static fs::path InitGoldenPath() noexcept
        {
            auto correctPermissions = [](const fs::path &path) {
                return (fs::status(path).permissions() & fs::perms::owner_all) != fs::perms::none;
            };

            try
            {
                auto goldenPathEnv = getenv("GOLDEN_PATH");

                fs::path path;
                if (goldenPathEnv)
                    path = fs::weakly_canonical(std::filesystem::path(goldenPathEnv));
                else
                    path = fs::weakly_canonical(std::filesystem::path(golden_GOLDEN_STORAGE_PATH));

                if (fs::exists(path))
                {
                    if (!fs::is_directory(path))
                    {
                        std::cout << "Golden path exists and is not a directory." << std::endl;
                        exit(-1);
                    }
                    // TODO not sure how is_directory handles symlinks
                    else if (correctPermissions(path))
                    {
                        return path;
                    }
                    else
                    {
                        std::cout << "Golden path exists and is a directory, but has the wrong permissions."
                                  << std::endl;
                        exit(-1);
                    }
                }
                // only create path if the parent path exists
                else if (fs::exists(path.parent_path()) && correctPermissions(path.parent_path()))
                {
                    // TODO might want something else on Windows
                    // as per this documentation attributes are not copied on Windows
                    // https://en.cppreference.com/w/cpp/filesystem/create_directory
                    //
                    // - create directory with parent path permissions
                    create_directory(path, path.parent_path());
                    return path;
                }
                else
                {
                    std::cout << "Golden parent path does not exist or has the wrong permissions" << std::endl;
                    exit(-1);
                }
            }
            catch (...)
            {
                std::cout << "Initializing golden storage path failed." << std::endl;
                exit(-1);
            }
        }

        inline static const fs::path GOLDEN_PATH = InitGoldenPath();
    };
} // namespace golden

#endif // GOLDEN_UTILITY_HPP
