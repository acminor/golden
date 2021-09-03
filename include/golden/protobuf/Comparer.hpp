//
// Created by austin on 8/2/21.
//

#ifndef GOLDEN_PROTOBUF_COMPARER_HPP
#define GOLDEN_PROTOBUF_COMPARER_HPP

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
            inline Comparer() : m_messageDifferencer(), m_fieldComparator()
            {
                m_fieldComparator.set_float_comparison(DefaultFieldComparator::APPROXIMATE);
                m_fieldComparator.set_treat_nan_as_equal(true); // TODO needs unittest
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
    } // namespace protobuf
} // namespace golden

#endif // GOLDEN_PROTOBUF_COMPARER_HPP
