/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#pragma once

#include "Definitions.hpp"

#include <set>

namespace dd {
struct Control {
  // make dd -> op you need -> polarized control, unsigned integer
  using Type = std::uint_fast8_t;

  QuantumRegister quantumRegister{};
  Type type = 1; // default value of a control level
};

inline bool operator<(const Control& lhs, const Control& rhs) {
  return lhs.quantumRegister < rhs.quantumRegister ||
         (lhs.quantumRegister == rhs.quantumRegister && lhs.type < rhs.type);
}

inline bool operator==(const Control& lhs, const Control& rhs) {
  return lhs.quantumRegister == rhs.quantumRegister && lhs.type == rhs.type;
}

inline bool operator!=(const Control& lhs, const Control& rhs) {
  return !(lhs == rhs);
}

// this allows a set of controls to be indexed by a quantum register, namely a
// qudit w/ d>=2
struct CompareControl {
  using is_transparent = void;

  inline bool operator()(const Control& lhs, const Control& rhs) const {
    return lhs < rhs;
  }

  inline bool operator()(QuantumRegister lhs, const Control& rhs) const {
    return lhs < rhs.quantumRegister;
  }

  inline bool operator()(const Control& lhs, QuantumRegister rhs) const {
    return lhs.quantumRegister < rhs;
  }
};
using Controls = std::set<Control, CompareControl>;

inline namespace literals {
// NOLINTNEXTLINE(google-runtime-int): Standard mandates usage of ULL
inline Control operator""_pc(unsigned long long int qreg) {
  return {static_cast<QuantumRegister>(qreg)};
}
} // namespace literals
} // namespace dd
