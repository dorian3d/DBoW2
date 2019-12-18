cmake_minimum_required(VERSION 3.12)
project(tdbow)

####################
## Option setting ##
####################

option(BUILD_TDBOW_PC_Demo "Build PC demo" ON)
option(BUILD_TDBOW_CV_Demo "Build CV demo" ON)
option(ABSOLUTE_LOG    "Log absolute path" ON)

if(NOT ABSOLUTE_LOG)
  add_definitions(-DTDBOW_RELATIVE_LOG)
endif()

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
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
find_package(catkin REQUIRED)

## Eigen and boost are required

# Manually resolve removed dependend packages
find_package(Eigen3 QUIET)
if (NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else ()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif ()

find_package(Boost REQUIRED COMPONENTS filesystem)
if(NOT (Boost_VERSION VERSION_LESS "106000"))
  add_definitions(-DFOUND_BOOST_1_60)
endif()
set(EXTRA_INCS ${Boost_INCLUDE_DIRS} ${EXTRA_INCS})
set(EXTRA_LIBS ${Boost_LIBRARIES}    ${EXTRA_LIBS})

## Setting the extra libraries

## QuickLZ lib is installed
set(QUICKLZ_INC libs/quicklz)
set(QUICKLZ_HDR libs/quicklz/quicklz.h)
set(QUICKLZ_SRC libs/quicklz/quicklz.c)
set(EXTRA_INCS ${QUICKLZ_INC} ${EXTRA_INCS})
set(EXTRA_HDRS ${QUICKLZ_HDR} ${EXTRA_HDRS})
set(EXTRA_SRCS ${QUICKLZ_SRC} ${EXTRA_SRCS})

## Yaml-cpp is required
set(EXTRA_LIBS yaml-cpp ${EXTRA_LIBS})

###################################
## catkin specific configuration ##
###################################
catkin_package(
    INCLUDE_DIRS include libs/quicklz
    LIBRARIES ${PROJECT_NAME}
)

link_libraries(
    ${catkin_LIBRARIES}
)

## Setting TDBoW context
set(TDBoW_INC include/TDBoW)
set(TDBoW_LIB pthread)
file(GLOB_RECURSE TDBoW_HDRS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${TDBoW_INC}/*.h
        ${TDBoW_INC}/*.hpp)
file(GLOB_RECURSE TDBoW_SRCS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        src/*.c
        src/*.cc
        src/*.cpp)

set(LIB_SHARED "SHARED")
if(WIN32)
  set(LIB_SHARED "STATIC")
endif(WIN32)
add_library(${PROJECT_NAME} ${LIB_SHARED} ${TDBoW_SRCS} ${EXTRA_SRCS})
include_directories(include ${EXTRA_INCS} ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${TDBoW_LIB} ${EXTRA_LIBS})

if(BUILD_TDBOW_CV_Demo)
  ## Setting context
  set(EXEC_NAME cv_demo)
  set(EXEC_TARGET_NAME ${PROJECT_NAME}-${EXEC_NAME}-exec)
  ## OpenCV required
  find_package(OpenCV REQUIRED)
  ## Build target
  add_executable(${EXEC_TARGET_NAME} demo/cv/demo.cpp)
  target_link_libraries(${EXEC_TARGET_NAME} ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(${EXEC_TARGET_NAME} PROPERTIES
          OUTPUT_NAME ${EXEC_NAME})
endif()

if(BUILD_TDBOW_PC_Demo)
  ## Setting context
  set(EXEC_NAME pc_demo)
  set(EXEC_TARGET_NAME ${PROJECT_NAME}-${EXEC_NAME}-exec)
  ## PCL required && OpenMP optional
  find_package(OpenMP QUIET)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DFOUND_OPENMP)
    message(STATUS "OpenMP Founded")
  endif()
  find_package(PCL 1.7.2 REQUIRED COMPONENTS common io keypoints features)
  add_definitions(${PCL_DEFINITIONS})
  ## Build target
  add_executable(${EXEC_TARGET_NAME} demo/pc/demo.cpp)
  target_include_directories(${EXEC_TARGET_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
  target_link_libraries(${EXEC_TARGET_NAME} ${PROJECT_NAME} ${PCL_LIBRARIES})
  set_target_properties(${EXEC_TARGET_NAME} PROPERTIES
          OUTPUT_NAME ${EXEC_NAME})
endif()

#############
## Install ##
#############
# Install all targets, headers by default and scripts and other files if specified (folders or files)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(FILES ${TDBoW_HDRS} ${QUICKLZ_HDR}
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
