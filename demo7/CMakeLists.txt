cmake_minimum_required(VERSION 3.16)

project(TestOpencv LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 OpenCV 路径
set(OpenCV_INCLUDE_DIRS "E:/OpenCV/build/include" "E:/OpenCV/build/include/opencv2")
set(OpenCV_LIB_DIR "E:/OpenCV/build/x64/vc16/lib")

# 查找 OpenCV 库文件
file(GLOB OpenCV_LIBS "${OpenCV_LIB_DIR}/*.lib")

# 添加 OpenCV 头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 Qt
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets)

# 添加可执行文件
add_executable(TestOpencv
  main.cpp
)

# 链接 OpenCV 库
target_link_libraries(TestOpencv ${OpenCV_LIBS})

# 链接 Qt 库
target_link_libraries(TestOpencv Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Widgets)

# 安装配置
include(GNUInstallDirs)
install(TARGETS TestOpencv
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
