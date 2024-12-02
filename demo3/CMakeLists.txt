cmake_minimum_required(VERSION 3.10)

project(tensorrt_win_project)

add_definitions(-std=c++14)

add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

set(CUDA_COMPILER_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
set(TENSORRT_PATH "C:/Users/zeng'li/Downloads/TensorRT-10.5.0.18.Windows.win10.cuda-12.6/TensorRT-10.5.0.18")

find_package(CUDA REQUIRED) 
include_directories(${CUDA_INCLUDE_DIRS}) 
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

include_directories(${TENSORRT_PATH}/include)
link_directories(${TENSORRT_PATH}/lib)

find_package(OpenCV REQUIRED
            CONFIG
            PATHS C:/opencv/opencv/build/x64/vc16/lib)
include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND) 
    include_directories(${OpenCV_INCLUDE_DIRS}) 
    link_directories(${OpenCV_LIBRARY_DIRS}) 
    add_definitions(${OpenCV_DEFINITIONS}) 
else() 
    message(FATAL_ERROR "OpenCV not found!") 
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable (tensorrt_win_project ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})

target_link_libraries(tensorrt_win_project nvinfer_10 nvinfer_plugin_10 cudart nvonnxparser_10 cudnn  ${OpenCV_LIBS})