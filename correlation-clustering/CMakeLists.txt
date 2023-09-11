cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
#set(CMAKE_BUILD_TYPE DEBUG)

project(cubic-multicut)
enable_testing()
#set(CMAKE_CXX_STANDARD 20)

include_directories(include)

# experiments
add_executable(correlation-clustering src/correlation-clustering.cxx)
target_include_directories(correlation-clustering PUBLIC include)

add_executable(correlation-clustering-classifier src/correlation-clustering-classifier.cxx)
target_include_directories(correlation-clustering-classifier PUBLIC include)

add_executable(correlation-clustering-subsets src/correlation-clustering-subsets.cxx)
target_include_directories(correlation-clustering-subsets PUBLIC include)

add_executable(accuracy-0-1 src/accuracy-0-1.cxx)
target_include_directories(accuracy-0-1 PUBLIC include)


# add_executable(experiment-constellations src/experiments/constellations.cxx)
# target_include_directories(experiment-constellations PUBLIC include)
