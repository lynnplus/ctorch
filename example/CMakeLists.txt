
cmake_minimum_required(VERSION 3.20)
project(detector)

find_package(OpenCV 4.5.0 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${LIB_NAME})