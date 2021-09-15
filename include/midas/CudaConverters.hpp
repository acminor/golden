//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_CUDACONVERTERS_HPP
#define GOLDEN_CUDACONVERTERS_HPP

#include "midas/Dim3Converter.hpp"
#include "midas/Float4Converter.hpp"
#include "midas/Int4Converter.hpp"
#include "midas/PrimitiveConverter.hpp"
#include "midas/VectorConverter.hpp"

#ifdef MIDAS_CUDA_MALLOC_INTEROP_ENABLED
#include "midas/MallocInteropArrayConverter.hpp"
#endif /* MIDAS_CUDA_MALLOC_INTEROP_ENABLED */

#endif // GOLDEN_CUDACONVERTERS_HPP
