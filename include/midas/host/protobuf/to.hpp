//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_TO_HPP
#define GOLDEN_TO_HPP

namespace midas::host::protobuf
{
    template <typename T, typename Function>
    void to(const T *mem, size_t numberOfElements, Function updateProtobufElement)
    {
        auto data = std::vector(mem, mem + numberOfElements);

        for (auto x : data)
            updateProtobufElement(x);
    }
} // namespace midas::host::protobuf

#endif // GOLDEN_TO_HPP
