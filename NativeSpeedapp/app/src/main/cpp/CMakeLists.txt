cmake_minimum_required(VERSION 3.10.2)
project("nativespeedapp")
set(CMAKE_CXX_STANDARD 14)

link_directories("${CMAKE_CURRENT_SOURCE_DIR}/frameworks/libraries")

add_library(native-lib SHARED native-lib.cpp)
find_library(log-lib log)

target_include_directories(native-lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/frameworks/includes")
target_link_libraries(native-lib ${log-lib} tflite torchscript ort )