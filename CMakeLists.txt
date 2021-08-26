cmake_minimum_required(VERSION 3.19)
project(golden VERSION 0.3.0)

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
       include/golden/protobuf/Tester.hpp
       include/golden/protobuf/Writer.hpp
       include/golden/silver/Silver.hpp
       include/golden/silver/SilverPlate.hpp
       include/golden/silver/protobuf/SilverPlate.hpp)
target_include_directories(golden INTERFACE ${Protobuf_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(golden INTERFACE ${Protobuf_LIBRARIES})

enable_testing()
add_subdirectory(./tests)