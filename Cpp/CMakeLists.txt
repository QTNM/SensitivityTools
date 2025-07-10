# - Basic CMake setup
# Check version meets your requirements
# Declare project, which will configure compiler for us
cmake_minimum_required(VERSION 3.20)
project(rfit)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# enable threading
list(APPEND CMAKE_CXX_FLAGS "-pthread -std=c++20 ${CMAKE_CXX_FLAGS}")

#Require ROOT, initially try finding previously installed root
find_package(ROOT REQUIRED COMPONENTS RooFit RooFitCore RooStats Minuit)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXX_FLAGS}")

# START
include_directories(include ${ROOT_INCLUDES})

ROOT_GENERATE_DICTIONARY(app_dict 
  RooKurieNHPdf.h 
  RooBDNormalHPdf.h 
  RooBDNormalSterilePdf.h
  RooBDInverseHPdf.h
  LINKDEF include/LinkDef.h)

# Build a dynamic library from our sources
add_library(rfit SHARED 
  app_dict.cxx
  src/RooKurieNHPdf.cxx
  src/RooBDNormalHPdf.cxx
  src/RooBDInverseHPdf.cxx
  src/RooBDNormalSterilePdf.cxx)
target_link_libraries(rfit ${ROOT_LIBRARIES})

#Executables
add_executable(firstapp.exe firstapp.cpp)
target_link_libraries(firstapp.exe 
  rfit ${ROOT_LIBRARIES})

add_executable(model.exe modeltrial.cpp)
target_link_libraries(model.exe 
  rfit ${ROOT_LIBRARIES})
