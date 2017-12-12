cmake_minimum_required(VERSION 2.8)
project(test-paddle)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CheckCXXCompilerFlag)

if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
endif()

message("PADDLEROOT: $ENV{PADDLEROOT}")
# ---[ Find include dir
find_path(PADDLE_INCLUDE_DIR NAMES "paddle/capi.h" HINTS "$ENV{PADDLEROOT}" PATHS "$ENV{PADDLEROOT}" PATH_SUFFIXES "include" )

message("PADDLE_INCLUDE: ${PADDLE_INCLUDE_DIR}")

find_library(PADDLE_LIBRARY
        NAMES paddle_capi_shared
        PATHS $ENV{PADDLEROOT}/lib)

message("PADDLE_LIBRARY: ${PADDLE_LIBRARY}")

if(PADDLE_LIBRARY)
  message(STATUS "Found Paddle (include: ${PADDLE_INCLUDE_DIR}, lib: ${PADDLE_LIBRARY}")
  include_directories(${PADDLE_INCLUDE_DIR})
else()
  message(FATAL " Intel PADDLE not located properly")
endif()

add_executable(test-paddle main.cpp)

add_definitions(-Wall)

target_link_libraries(test-paddle ${PADDLE_LIBRARY} )
