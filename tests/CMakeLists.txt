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
        PRIVATE ${GTEST_INCLUDE_DIR} ../include ${CMAKE_BINARY_DIR}/tests ${CMAKE_BINARY_DIR}/include)
target_link_libraries(golden_tests PRIVATE gtest gtest_main)
target_include_directories(golden_tests PRIVATE ${Protobuf_INCLUDE_DIRS})
target_link_libraries(golden_tests PRIVATE ${Protobuf_LIBRARIES})