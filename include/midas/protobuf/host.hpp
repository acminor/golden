//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_PROTOBUF_HOST_HPP
#define GOLDEN_MIDAS_PROTOBUF_HOST_HPP

namespace midias::protobuf::host
{
    namespace detail
    {
    }

    template <typename ElementType, typename T> std::vector<ElementType> protobufToVector(T snapshotField)
        {
        std::vector<ElementType> host_vector(snapshotField.size());

        int i = 0;
        for (const auto &x : snapshotField)
        {
            if constexpr (ProtobufFieldInformation<T, ElementType>::HasConversionFunction)
            host_vector[i++] = conversion::protobufToCuda(x);
            else
                host_vector[i++] = x;
        }

        return std::move(host_vector);
        }

} // namespace midias::protobuf::host

#endif // GOLDEN_MIDAS_PROTOBUF_HOST_HPP
