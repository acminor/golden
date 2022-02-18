//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_COMPARER_HPP
#define GOLDEN_PROTOBUF_COMPARER_HPP

#include <cstdlib>

#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/message_differencer.h"

namespace golden
{
    namespace protobuf
    {
        using namespace google::protobuf::util;

        class Comparer
        {
          public:
            inline Comparer() : m_messageDifferencer(), m_fieldComparator(), m_reporter()
            {
                m_fieldComparator.set_float_comparison(DefaultFieldComparator::APPROXIMATE);
                m_fieldComparator.set_treat_nan_as_equal(true); // TODO needs unittest
                m_messageDifferencer.set_field_comparator(&m_fieldComparator);
                m_messageDifferencer.set_message_field_comparison(MessageDifferencer::EQUAL);
                m_messageDifferencer.set_scope(MessageDifferencer::FULL);

                if (getenv("GOLDEN_PROTO_REPORT_STRING"))
                    m_messageDifferencer.ReportDifferencesToString(&m_reporter);
            }

            template <typename T>
            inline int compare(T a, T b)
            {
                m_reporter.clear();
                auto equal = m_messageDifferencer.Compare(a, b);

                if (equal)
                {
                    return 0;
                }
                else
                {
                    if (getenv("GOLDEN_PROTO_REPORT_STRING"))
                        std::cout << m_reporter << std::endl;
                    return -1;
                }
            }

          private:
            DefaultFieldComparator m_fieldComparator;
            MessageDifferencer m_messageDifferencer;
            std::string m_reporter;
        };
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_COMPARER_HPP
