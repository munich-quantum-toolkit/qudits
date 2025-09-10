# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_QUDITS_TARGET_NAME})
  # collect headers
  file(GLOB_RECURSE MQT_QUDITS_HEADERS ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/*.hpp)

  add_library(${MQT_QUDITS_TARGET_NAME} INTERFACE ${MQT_QUDITS_HEADERS})

  # set include directories
  target_include_directories(${MQT_QUDITS_TARGET_NAME}
                             INTERFACE $<BUILD_INTERFACE:${MQT_QUDITS_INCLUDE_BUILD_DIR}>)

  # add options and warnings to the library
  target_link_libraries(${MQT_QUDITS_TARGET_NAME} INTERFACE MQT::ProjectOptions
                                                            MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::Qudits ALIAS ${MQT_QUDITS_TARGET_NAME})

  target_compile_features(${MQT_QUDITS_TARGET_NAME} INTERFACE cxx_std_20)
endif()
