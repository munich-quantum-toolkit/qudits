# set required cmake version
cmake_minimum_required(VERSION 3.19...3.28)

# project definition
project(
  mqt-qudits
  LANGUAGES CXX
  VERSION 0.1.0
  DESCRIPTION "MQT Qudits - A framework for mixed-dimensional qudit quantum computing")

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/PackageAddTest.cmake)
include(cmake/Cache.cmake)

option(BUILD_MQT_QUDITS_BINDINGS "Build the MQT Qudits Python bindings" OFF)
if(BUILD_MQT_QUDITS_BINDINGS)
  # ensure that the BINDINGS option is set
  set(BINDINGS
      ON
      CACHE INTERNAL "Enable settings related to Python bindings")
  # Some common settings for finding Python
  set(Python_FIND_VIRTUALENV
      FIRST
      CACHE STRING "Give precedence to virtualenvs when searching for Python")
  set(Python_FIND_FRAMEWORK
      LAST
      CACHE STRING "Prefer Brew/Conda to Apple framework Python")
  set(Python_ARTIFACTS_INTERACTIVE
      ON
      CACHE BOOL "Prevent multiple searches for Python and instead cache the results.")

  # top-level call to find Python
  find_package(
    Python 3.8 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_QUDITS_MASTER_PROJECT ON)
else()
  set(MQT_QUDITS_MASTER_PROJECT OFF)
endif()
option(BUILD_MQT_QUDITS_TESTS "Also build tests for the MQT MiSiM project"
       ${MQT_QUDITS_MASTER_PROJECT})

include(cmake/ExternalDependencies.cmake)

# set the include directory for the build tree
set(MQT_QUDITS_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# add main library code
add_subdirectory(src)

# add test code
if(BUILD_MQT_QUDITS_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
