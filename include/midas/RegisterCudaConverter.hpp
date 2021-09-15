//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_REGISTERCUDACONVERTER_HPP
#define GOLDEN_REGISTERCUDACONVERTER_HPP

#define RegisterCudaConverter2(CONVERTER_TYPE, CONVERTER_NAME)                                                         \
    namespace converters                                                                                               \
    {                                                                                                                  \
        CONVERTER_TYPE CONVERTER_NAME;                                                                                 \
    }

#define RegisterCudaConverter1(CONVERTER_TYPE_AND_NAME)                                                                \
    namespace converters                                                                                               \
    {                                                                                                                  \
        CONVERTER_TYPE_AND_NAME CONVERTER_TYPE_AND_NAME;                                                               \
    }

#endif // GOLDEN_REGISTERCUDACONVERTER_HPP
