cmake_minimum_required(VERSION 3.10)

project(opencv_win_project)

set(OpenCV_DIR "C:/opencv/build") 

# 查找OpenCV库
find_package(OpenCV REQUIRED)

 if(Opencv_FOUND)
     message(STATUS "The Opecv lib is found!") 
 endif()

# 添加可执行文件
add_executable(opencv_win_project main.cpp)

# 链接OpenCV库
target_link_libraries(opencv_win_project ${OpenCV_LIBS})


