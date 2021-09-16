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
        ${CMAKE_BINARY_DIR}/include/midas/ProtobufSupport.pb.cc
        ${CMAKE_BINARY_DIR}/include/midas/ProtobufSupport.pb.h)
add_custom_target(
        protobuf_support_gen
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/generate-protobuf-support.sh
        ${CMAKE_SOURCE_DIR}/include/midas/ProtobufSupport.proto
        ${CMAKE_BINARY_DIR}/include
)
add_dependencies(protobuf_support protobuf_support_gen)

add_executable(midas_tests)
target_sources(midas_tests
        PRIVATE
        MidasTests.cu)
target_include_directories(midas_tests PRIVATE ${CMAKE_BINARY_DIR}/include)
target_link_libraries(midas_tests PRIVATE midas gtest gtest_main protobuf_support)