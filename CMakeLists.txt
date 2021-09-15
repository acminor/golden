cmake_minimum_required(VERSION 3.19)
project(golden VERSION 0.4.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf)

add_library(golden INTERFACE include/Golden.hpp)
target_sources(golden
        INTERFACE
        include/Golden.hpp
        include/golden/Config.hpp
        include/golden/Tester.hpp
        include/golden/Utility.hpp
        include/golden/protobuf/Comparer.hpp
        include/golden/protobuf/Reader.hpp
        include/golden/protobuf/ReaderZlib.hpp
        include/golden/protobuf/Tester.hpp
        include/golden/protobuf/Writer.hpp
        include/golden/protobuf/WriterZlib.hpp
        include/golden/protobuf/Zlib.h
        include/golden/silver/Silver.hpp
        include/golden/silver/SilverPlate.hpp
        include/golden/silver/protobuf/SilverPlate.hpp)
target_include_directories(golden INTERFACE ${Protobuf_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(golden INTERFACE ${Protobuf_LIBRARIES})

add_library(midas INTERFACE include/midas/midas.hpp)
target_sources(midas
        INTERFACE
        include/midas/cuda/utility.hpp
        include/midas/cuda/MallocInterop.hpp
        include/midas/MallocInteropArrayConverter.hpp
        include/midas/VectorConverter.hpp
        include/midas/Dim3Converter.hpp
        include/midas/Float4Converter.hpp
        include/midas/Int4Converter.hpp
        include/midas/RegisterCudaConverter.hpp
        include/midas/CudaConverters.hpp
        include/midas/Converters.hpp)
target_include_directories(midas INTERFACE ${Protobuf_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(midas INTERFACE ${Protobuf_LIBRARIES})

enable_testing()
add_subdirectory(./tests)