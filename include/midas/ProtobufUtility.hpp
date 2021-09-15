//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_PROTOBUFUTILITY_HPP
#define GOLDEN_PROTOBUFUTILITY_HPP

namespace midas::protobuf
{
    struct NoConverter;
    template <typename ProtobufField, typename ElementTypeIn, typename Converter = NoConverter>
    struct ProtobufFieldInformation
    {
      private:
        inline static constexpr bool IsEquivalentBase()
        {
            if constexpr (std::is_convertible_v<ElementType, ElementTypeIn>)
                return true;
            else if constexpr (!std::is_same_v<Converter, NoConverter>)
                return true;

            return false;
        }

      public:
        using FieldType = ProtobufField;
        using ElementType = decltype(((FieldType *)0)->Get(0));

        inline static constexpr bool IsEquivalent = IsEquivalentBase();
        using EnableIfEquivalent = std::enable_if_t<IsEquivalent, bool>;
    };

    struct NoConverter
    {
    };

    template <typename ElementType, typename T, typename Converter = NoConverter,
              typename ProtobufFieldInformation<T, ElementType>::EnableIfEquivalent = true>
    std::vector<ElementType> &&protobufToVector(T snapshotField, Converter converter = {})
    {
        std::vector<ElementType> host_vector(snapshotField.size());

        int i = 0;
        for (const auto &x : snapshotField)
        {
            if constexpr (!std::is_same_v<Converter, NoConverter>)
            {
                ElementType temp;
                converter.Deserialize(x, &temp);
                host_vector[i++] = temp;
            }
            else
            {
                host_vector[i++] = x;
            }
        }

        return std::move(host_vector);
    }
} // namespace midas::protobuf

#endif // GOLDEN_PROTOBUFUTILITY_HPP
