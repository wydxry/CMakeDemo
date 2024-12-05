cmake_minimum_required(VERSION 3.10)

project(onnx_tensorrt_win_project)

add_definitions(-std=c++14)

add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

set(CUDA_COMPILER_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
set(TENSORRT_PATH "C:/Users/zeng'li/Downloads/TensorRT-10.5.0.18.Windows.win10.cuda-12.6/TensorRT-10.5.0.18")
set(ONNX_PATH "C:/Users/zeng'li/Downloads/onnxruntime-win-x64-gpu-cuda12-1.18.0/onnxruntime-win-x64-gpu-1.18.0")
set(CUDNN_PATH "C:/Users/zeng'li/Downloads/cudnn-windows-x86_64-8.9.7.29_cuda12-archive/cudnn-windows-x86_64-8.9.7.29_cuda12-archive/")
set(ONNX_CORE_PATH "C:/Users/zeng'li/Downloads/onnxruntime-win-x64-gpu-cuda12-1.18.0/onnxruntime-win-x64-gpu-1.18.0/include/core/providers/cuda")

find_package(CUDA REQUIRED) 
include_directories(${CUDA_INCLUDE_DIRS}) 
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

include_directories(${TENSORRT_PATH}/include)
link_directories(${TENSORRT_PATH}/lib)

include_directories(${ONNX_CORE_PATH}/include)
include_directories(${ONNX_PATH}/include)
link_directories(${ONNX_PATH}/lib)

include_directories(${CUDNN_PATH}/include)
link_directories(${CUDNN_PATH}/lib)

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

add_executable (onnx_tensorrt_win_project ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})

target_link_libraries(onnx_tensorrt_win_project nvinfer_10 nvinfer_plugin_10 cudart nvonnxparser_10 cudnn  
					onnxruntime onnxruntime_providers_cuda onnxruntime_providers_shared onnxruntime_providers_tensorrt ${OpenCV_LIBS})