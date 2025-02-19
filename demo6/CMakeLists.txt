cmake_minimum_required(VERSION 3.16)

project(TestHidapi LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

# 设置HIDAPI的路径
set(HIDAPI_INCLUDE_DIR "thirdparty/Hidapi/include/")
set(HIDAPI_LIB_DIR "thirdparty/Hidapi/lib/")


add_executable(TestHidapi
  main.cpp
)

# 添加包含目录
target_include_directories(TestHidapi PRIVATE ${HIDAPI_INCLUDE_DIR})

# 添加库目录
target_link_directories(TestHidapi PRIVATE ${HIDAPI_LIB_DIR})

target_link_libraries(TestHidapi HIDAPI setupapi.lib Qt${QT_VERSION_MAJOR}::Core)

include(GNUInstallDirs)
install(TARGETS TestHidapi
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
