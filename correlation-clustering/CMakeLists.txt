cmake_minimum_required(VERSION 3.10)

project(correlation-clustering-of-bird-sounds LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include_directories(include)

# experiments
add_executable(correlation-clustering src/correlation-clustering.cxx)
add_executable(correlation-clustering-classifier src/correlation-clustering-classifier.cxx)
add_executable(correlation-clustering-subsets src/correlation-clustering-subsets.cxx)
add_executable(accuracy-0-1 src/accuracy-0-1.cxx)
