cmake_minimum_required (VERSION 2.8.11)
project (US-famli)


option(KretzConverter "Build the Kretz converter" OFF)

find_package(ITK)
include(${ITK_USE_FILE})

find_package(SlicerExecutionModel)
include(${SlicerExecutionModel_USE_FILE})

option(USE_VTK "Use VTK to build the flip image tool" OFF)

if(USE_VTK)
	find_package(VTK)
	include(${VTK_USE_FILE})
endif()

add_subdirectory (src)