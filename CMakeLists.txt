cmake_minimum_required (VERSION 2.8.11)
project (US-famli)


find_package(ITK)
include(${ITK_USE_FILE})

find_package(SlicerExecutionModel)
include(${SlicerExecutionModel_USE_FILE})

add_subdirectory (src)