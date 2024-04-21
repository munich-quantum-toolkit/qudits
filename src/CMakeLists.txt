set(MQT_QUDITS_TARGET_NAME "mqt-qudits")

if(NOT TARGET project_warnings)
  # Use the warnings specified in CompilerWarnings.cmake
  add_library(project_warnings INTERFACE)

  # Standard compiler warnings
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
  set_project_warnings(project_warnings)

  # Add MQT alias
  add_library(MQT::ProjectWarnings ALIAS project_warnings)
  set_target_properties(project_warnings PROPERTIES EXPORT_NAME ProjectWarnings)
endif()

if(NOT TARGET project_options)
  # Use the options specified in CompilerOptions.cmake
  add_library(project_options INTERFACE)

  # Standard compiler options
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerOptions.cmake)
  enable_project_options(project_options)

  # Sanitizer options if supported by compiler
  include(${PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake)
  enable_sanitizers(project_options)

  # Add MQT alias
  add_library(MQT::ProjectOptions ALIAS project_options)
  set_target_properties(project_options PROPERTIES EXPORT_NAME ProjectOptions)
endif()

if(NOT TARGET ${MQT_QUDITS_TARGET_NAME})
  add_library(
    ${MQT_QUDITS_TARGET_NAME}
    INTERFACE
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/Complex.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/ComplexCache.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/ComplexNumbers.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/ComplexTable.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/ComplexValue.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/ComputeTable.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/Control.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/Definitions.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/Edge.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/GateMatrixDefinitions.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/MDDPackage.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/UnaryComputeTable.hpp
    ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/UniqueTable.hpp)

  # set include directories
  target_include_directories(${MQT_QUDITS_TARGET_NAME}
                             INTERFACE $<BUILD_INTERFACE:${MQT_QUDITS_INCLUDE_BUILD_DIR}>)

  # add options and warnings to the library
  target_link_libraries(${MQT_QUDITS_TARGET_NAME} INTERFACE MQT::ProjectOptions
                                                            MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::Qudits ALIAS ${MQT_QUDITS_TARGET_NAME})
endif()

if(BUILD_MQT_QUDITS_BINDINGS)
  add_subdirectory(python)
endif()
