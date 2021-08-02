cmake_minimum_required(VERSION 3.10.2)

project("mobileneuralnetwork")

# ADD MNN TO OUR PROJECT
set(mnn_lib_DIR ${CMAKE_SOURCE_DIR}/../jni)
include_directories(${CMAKE_SOURCE_DIR}/includes)

add_library(mnn SHARED IMPORTED)
set_target_properties(mnn PROPERTIES IMPORTED_LOCATION ${mnn_lib_DIR}/${ANDROID_ABI}/libMNN.so)

add_library(mnncore SHARED mnn_native.cpp)
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)
target_link_libraries(mnncore ${log-lib} ${jnigraphics-lib} mnn)