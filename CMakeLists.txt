cmake_minimum_required(VERSION 3.19)
project(golden VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)

set(protobuf_BUILD_TESTS OFF)
#add_subdirectory(./libs/protobuf/cmake)
find_package(Protobuf)

set(golden_GOLDEN_STORAGE_PATH "${CMAKE_SOURCE_DIR}/tests/GOLDEN/" CACHE PATH "Path where golds are stored")

configure_file(
        "${PROJECT_SOURCE_DIR}/include/Config.hpp.in"
        "${PROJECT_BINARY_DIR}/include/Config.hpp"
)

add_library(golden INTERFACE include/Tester.hpp)
#target_include_directories(golden INTERFACE libprotobuf)
#target_link_libraries(golden INTERFACE libprotobuf)
target_include_directories(golden INTERFACE ${Protobuf_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/include)
target_link_libraries(golden INTERFACE ${Protobuf_LIBRARIES})

enable_testing()
add_subdirectory(./tests)