add_subdirectory(libs/googletest)

protobuf_generate_cpp(golden_tests_SRCS golden_tests_INCLUDES GoldenTests.proto)

add_executable(golden_tests)
target_sources(golden_tests
        PRIVATE
        GoldenTests.cpp
        ProtobufSilver.cpp
        ProtobufComparerTests.cpp
        ProtobufReaderTests.cpp
        ProtobufWriterTests.cpp
        ${golden_tests_SRCS}
        ${golden_tests_INCLUDES})
target_include_directories(golden_tests
        PRIVATE ${GTEST_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/tests)
target_link_libraries(golden_tests PRIVATE golden gtest gtest_main)

add_library(protobuf_support)
target_link_libraries(protobuf_support PRIVATE protobuf)
target_sources(protobuf_support PUBLIC
        ${CMAKE_BINARY_DIR}/tests/midas/midas/ProtobufSupport.pb.cc
        ${CMAKE_BINARY_DIR}/tests/midas/midas/ProtobufSupport.pb.h
        ${CMAKE_BINARY_DIR}/tests/midas/MidasTests.pb.cc
        ${CMAKE_BINARY_DIR}/tests/midas/MidasTests.pb.h)
target_include_directories(protobuf_support PUBLIC
        ${CMAKE_BINARY_DIR}/tests/midas)
add_custom_target(
        protobuf_support_gen
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/generate-midas-tests-protobufs.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
)
set_property(SOURCE ${CMAKE_BINARY_DIR}/tests/midas/midas/ProtobufSupport.pb.cc PROPERTY GENERATED 1)
set_property(SOURCE ${CMAKE_BINARY_DIR}/tests/midas/midas/ProtobufSupport.pb.h PROPERTY GENERATED 1)
set_property(SOURCE ${CMAKE_BINARY_DIR}/tests/midas/MidasTests.pb.cc PROPERTY GENERATED 1)
set_property(SOURCE ${CMAKE_BINARY_DIR}/tests/midas/MidasTests.pb.h PROPERTY GENERATED 1)
add_dependencies(protobuf_support protobuf_support_gen)

#add_executable(midas_cuda_tests)
#target_sources(midas_cuda_tests
#        PRIVATE
#        MidasTests.cu)
#target_include_directories(midas_cuda_tests PRIVATE ${CMAKE_BINARY_DIR}/include)
#target_link_libraries(midas_cuda_tests PRIVATE midas gtest gtest_main protobuf_support)

add_executable(midas_opencl_tests)
target_sources(midas_opencl_tests
        PRIVATE
        midas/opencl/Driver.cpp)
target_include_directories(midas_opencl_tests
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests/midas)
target_link_libraries(midas_opencl_tests PRIVATE midas gtest gtest_main protobuf_support OpenCL)

add_executable(midas_cuda_tests)
target_sources(midas_cuda_tests
        PRIVATE
        midas/cuda/Driver.cu)
target_include_directories(midas_cuda_tests
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests/midas)
target_link_libraries(midas_cuda_tests PRIVATE midas gtest gtest_main protobuf_support)

find_package(GTest)
gtest_discover_tests(golden_tests TEST_PREFIX Golden:)
# TODO could make a test for failing the different types of path errors in the code
# - this is the happy path test only
gtest_discover_tests(golden_tests TEST_PREFIX Golden:EnvironmentGoldenPath:
        PROPERTIES ENVIRONMENT GOLDEN_PATH=${CMAKE_BINARY_DIR}/tests/golden-env-var-path-test)
gtest_discover_tests(midas_opencl_tests TEST_PREFIX Midas:OpenCL:)
gtest_discover_tests(midas_cuda_tests TEST_PREFIX Midas:Cuda:)
