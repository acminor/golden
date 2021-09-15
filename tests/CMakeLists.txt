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

protobuf_generate_cpp(midas_SRCS midas_INCLUDES ${CMAKE_SOURCE_DIR}/include/midas/ProtobufSupport.proto)
add_executable(midas_tests)
target_sources(midas_tests
        PRIVATE
        MidasTests.cu
        ${midas_SRCS}
        ${midas_INCLUDES})
target_link_libraries(midas_tests PRIVATE midas gtest gtest_main)