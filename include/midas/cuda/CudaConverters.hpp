//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_MIDAS_CUDACONVERTERS_HPP
#define GOLDEN_MIDAS_CUDACONVERTERS_HPP

#include <midas/cuda/Dim3Converter.hpp>
#include <midas/cuda/Float4Converter.hpp>
#include <midas/cuda/Int4Converter.hpp>
#include <midas/cuda/PrimitiveConverter.hpp>
#include <midas/cuda/VectorConverter.hpp>

#ifdef MIDAS_CUDA_MALLOC_INTEROP_ENABLED
#include <midas/cuda/MallocInteropArrayConverter.hpp>
#endif /* MIDAS_CUDA_MALLOC_INTEROP_ENABLED */

#endif // GOLDEN_MIDAS_CUDACONVERTERS_HPP
