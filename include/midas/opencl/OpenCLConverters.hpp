//
// Created by acminor on 9/7/21.
//

#pragma once

#include <midas/opencl/Dim3Converter.hpp>
#include <midas/opencl/Float2Converter.hpp>
#include <midas/opencl/Float3Converter.hpp>
#include <midas/opencl/Float4Converter.hpp>
#include <midas/opencl/Int2Converter.hpp>
#include <midas/opencl/Int4Converter.hpp>
#include <midas/opencl/PrimitiveConverter.hpp>
//#include <midas/opencl/StringConverter.hpp>
#include <midas/opencl/VectorConverter.hpp>

#ifdef MIDAS_OPENCL_MALLOC_INTEROP_ENABLED
#include <midas/opencl/MallocInteropArrayConverter.hpp>
//#include <midas/opencl/MallocInteropStringConverter.hpp>
#endif /* MIDAS_CUDA_MALLOC_INTEROP_ENABLED */