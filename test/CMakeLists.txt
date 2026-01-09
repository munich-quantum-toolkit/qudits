# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add unit tests
package_add_test(mqt-qudits-test MQT::Qudits test_pkg.cpp)

add_executable(mqt-qudits-example main.cpp)
target_link_libraries(mqt-qudits-example PRIVATE MQT::Qudits)
set_target_properties(mqt-qudits-example PROPERTIES FOLDER tests)

add_executable(mqt-qudits-collect data_collect.cpp)
target_link_libraries(mqt-qudits-collect PRIVATE MQT::Qudits)
set_target_properties(mqt-qudits-collect PROPERTIES FOLDER tests)
