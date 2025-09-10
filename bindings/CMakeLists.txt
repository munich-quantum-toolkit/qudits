# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mqt_python_binding(
  QUDITS
  ${MQT_QUDITS_TARGET_NAME}-bindings
  bindings.cpp
  MODULE_NAME
  _qudits
  INSTALL_DIR
  .
  LINK_LIBS
  MQT::Qudits)

install(
  TARGETS ${MQT_QUDITS_TARGET_NAME}-bindings
  DESTINATION .
  COMPONENT ${MQT_QUDITS_TARGET_NAME}_Python)
