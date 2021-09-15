//
// Created by acminor on 9/1/21.
//

#ifndef GOLDEN_MIDIAS_HPP
#define GOLDEN_MIDIAS_HPP

#include <type_traits>
#include <utility>
#include <vector>

#include "mem.h"

namespace midias
{

    template <typename ProtobufField, typename ElementTypeIn> struct ProtobufFieldInformation
    {
      private:
        inline static constexpr bool IsEquivalentBase()
        {
            if constexpr (std::is_convertible_v<ElementType, ElementTypeIn>)
                return true;
            else if constexpr (HasConversionFunction)
                return true;

            return false;
        }

      public:
        using FieldType = ProtobufField;
        using ElementType = decltype(((FieldType *)0)->Get(0));
        inline static constexpr bool HasConversionFunction =
            std::is_invocable_r_v<ElementTypeIn, decltype(conversion::protobufToCuda), ElementType>;

        inline static constexpr bool IsEquivalent = IsEquivalentBase();
        using EnableIfEquivalent = std::enable_if_t<IsEquivalent, bool>;
    };
} // namespace midias

#endif // GOLDEN_MIDIAS_HPP
