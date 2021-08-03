//
// Created by austin on 7/30/21.
//

#ifndef GOLDEN_TESTER_HPP
#define GOLDEN_TESTER_HPP

#include "golden/Utility.hpp"

namespace golden
{
    template <typename Comparer, typename Reader, typename Writer> class TesterBase
    {
      public:
        inline TesterBase() : m_writer(), m_comparer(), m_reader()
        {
        }

        template <typename GoldenKey>
        inline GoldenResult _validate(GoldenKey key, const typename GoldenKey::MessageType result)
        {
            auto output = GoldenResult();
            if (GoldenUtility::GoldenExists(key))
            {
                typename GoldenKey::MessageType golden;
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

    template <typename TesterBase> class Tester
    {
      public:
        inline Tester() : m_testerBase()
        {
        }

        /**
         * Validates that two fields are equal
         */
        template <typename GoldenKey>
        inline GoldenResult validate(GoldenKey key, const typename GoldenKey::MessageType result)
        {
            return m_testerBase._validate(key, result);
        }

        /**
         * Require operates like validate but throws an exception.
         */
        template <typename GoldenKey> inline void require(GoldenKey key, const typename GoldenKey::MessageType result)
        {
            auto compareResult = m_testerBase._validate(key, result);

            if (!compareResult.isSuccess())
                throw "TODO Implement Require Execption";
        }

      private:
        TesterBase m_testerBase;
    };
} // namespace golden

#endif // GOLDEN_TESTER_HPP
