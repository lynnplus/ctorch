cmake_minimum_required(VERSION 3.20)
project(ctorch)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(LIB_NAME ctorch)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

#set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Ob0 /Od /RTC1")

option(BUILD_WITH_EXAMPLE "Build detector example" ON)
option(BUILD_SHARD_LIB "Build shared library" OFF)

find_package(Torch REQUIRED)

FILE(GLOB_RECURSE sourceFiles "source/*.cpp")

if (${BUILD_SHARD_LIB})
    add_compile_definitions(BUILDING_DLL)
    add_library(${LIB_NAME} SHARED ${sourceFiles})
else()
    add_compile_definitions(BUILDING_STATIC)
    add_library(${LIB_NAME} STATIC ${sourceFiles})
endif ()

target_include_directories(${LIB_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${LIB_NAME} ${TORCH_LIBRARIES})

if (${BUILD_WITH_EXAMPLE})
    add_subdirectory(example)
endif ()
