cmake_minimum_required(VERSION 2.8)
project(TDBoW)
include(ExternalProject)

####################
## Option setting ##
####################

option(BUILD_TDBoW      "Build TDBoW"   ON)
option(BUILD_PC_Demo    "Build PC demo" OFF)
option(BUILD_CV_Demo    "Build CV demo" ON)

## Compile as C++11
add_compile_options(-std=c++11)
add_definitions(-DPKG_DIR="${CMAKE_CURRENT_LIST_DIR}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
          "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()

## uncomment below when develop new utility or framework
#SET(CMAKE_CXX_FLAGS "-O0 -g ${CMAKE_CXX_FLAGS}")
## comment below when develop new utility or framework
SET(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")

###################
## find packages ##
###################

## Eigen and boost are required
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
if(NOT (Boost_VERSION VERSION_LESS "106000"))
  add_definitions(-DFOUND_BOOST_1_60)
endif()
set(EXTRA_INCS ${Boost_INCLUDE_DIRS} ${EXTRA_INCS})
set(EXTRA_LIBS ${Boost_LIBRARIES}    ${EXTRA_LIBS})

## QuickLZ lib is installed
set(QUICKLZ_INC libs/quicklz)
set(QUICKLZ_HDR libs/quicklz/quicklz.h)
set(QUICKLZ_SRC libs/quicklz/quicklz.c)
set(EXTRA_INCS ${QUICKLZ_INC} ${EXTRA_INCS})
set(EXTRA_HDRS ${QUICKLZ_HDR} ${EXTRA_HDRS})
set(EXTRA_SRCS ${QUICKLZ_SRC} ${EXTRA_SRCS})

## Yaml-cpp is required
set(EXTRA_LIBS yaml-cpp ${EXTRA_LIBS})

## Setting the extra libraries

## Setting TDBoW context
set(TDBoW_INC include/TDBoW)
set(TDBoW_LIB pthread yaml-cpp)
file(GLOB_RECURSE TDBoW_HDRS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${TDBoW_INC}/*.h
        ${TDBoW_INC}/*.hpp)
file(GLOB_RECURSE TDBoW_SRCS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        src/*.c
        src/*.cc
        src/*.cpp)
set(HDRS ${TDBoW_HDRS} ${EXTRA_HDRS})
set(SRCS ${TDBoW_SRCS} ${EXTRA_SRCS})

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

if(BUILD_TDBoW)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  include_directories(${TDBoW_INC} ${EXTRA_INCS})
  target_link_libraries(${PROJECT_NAME} ${TDBoW_LIB} ${EXTRA_LIBS})
endif()

if(BUILD_CV_Demo)
  find_package(OpenCV REQUIRED)
  add_executable(cv_demo demo/cv/demo.cpp)
  target_link_libraries(cv_demo ${PROJECT_NAME} ${OpenCV_LIBS})
  message(STATUS "`cv_demo` should be generated in ${PROJECT_BINARY_DIR}")
  file(COPY demo/cv/images DESTINATION ${PROJECT_BINARY_DIR}/)
endif()

configure_file(src/TDBoW.cmake.in
  "${PROJECT_BINARY_DIR}/TDBoWConfig.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_TDBoW)
  install(DIRECTORY ${TDBoW_INC} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TDBoWConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/TDBoWConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/TDBoW/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)
