//
// Created by austin on 8/3/21.
//

#ifndef GOLDEN_SILVER_SILVERPLATE_HPP
#define GOLDEN_SILVER_SILVERPLATE_HPP

#include "google/protobuf/message.h"

namespace golden
{
    namespace silver
    {
        template <typename Reader, typename Writer> class SilverPlateBase
        {
          public:
            SilverPlateBase() : m_reader(), m_writer()
            {
            }

            template <typename GoldenKey, typename Message,
                      std::enable_if_t<std::is_base_of<google::protobuf::Message, Message>::value, bool> = true>
            void SilverBase(GoldenKey key, Message message)
            {
                auto writer = Writer();
                writer.write(key, message);
            }

            template <
                typename GoldenKey, typename MessageFn,
                std::enable_if_t<std::is_invocable<MessageFn>::value, bool> = true,
                std::enable_if_t<std::is_base_of<google::protobuf::Message, std::invoke_result_t<MessageFn>>::value,
                                 bool> = true>
            void SilverBase(GoldenKey key, MessageFn messageFn)
            {
                SilverBase(key, messageFn());
            }

            // TODO needs testing
            template <
                typename GoldenKey, typename MessageFn,
                std::enable_if_t<std::is_invocable<MessageFn, typename GoldenKey::MessageType &>::value, bool> = true>
            void SilverBase(GoldenKey key, MessageFn messageFn)
            {
                typename GoldenKey::MessageType message;
                messageFn(message);
                SilverBase(key, message);
            }

            template <typename GoldenKey, typename RecoveryFn,
                      std::enable_if_t<std::is_invocable_v<RecoveryFn, typename GoldenKey::MessageType>, bool> = true>
            void DesilverBase(GoldenKey key, RecoveryFn recoveryFn)
            {
                auto reader = Reader();
                typename GoldenKey::MessageType message;

                reader.read(key, message);
                recoveryFn(message);
            }

          private:
            Reader m_reader;
            Writer m_writer;
        };

        template <typename SilverPlateBase> class SilverPlate
        {
          public:
            SilverPlate() : m_base()
            {
            }

            template <typename GoldenKey, typename MessageOrMessageFn>
            void Silver(GoldenKey key, MessageOrMessageFn message)
            {
                m_base.SilverBase(key, message);
            }

            template <typename GoldenKey, typename RecoveryFn> void Desilver(GoldenKey key, RecoveryFn recoveryFn)
            {
                m_base.DesilverBase(key, recoveryFn);
            }

          private:
            SilverPlateBase m_base;
        };
    } // namespace silver
} // namespace golden

#endif // GOLDEN_SILVER_SILVERPLATE_HPP
