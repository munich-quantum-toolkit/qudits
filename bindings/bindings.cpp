/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Control.hpp"
#include "dd/Definitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/MDDPackage.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>
#include <pybind11/cast.h>
#include <pybind11/complex.h> // NOLINT(misc-include-cleaner)
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

namespace {

using Instruction = std::tuple<std::string, bool, std::vector<int>, std::string,
                               std::vector<int>, py::object,
                               std::tuple<std::vector<dd::QuantumRegister>,
                                          std::vector<dd::Control::Type>>>;
using Circuit = std::vector<Instruction>;
using Circuit_info = std::tuple<unsigned int, std::vector<size_t>, Circuit>;

using Noise = std::tuple<double, double>;
using NoiseType = std::map<std::variant<std::string, std::vector<int>>, Noise>;
using NoiseModel = std::map<std::string, NoiseType>;
using CVec = std::vector<std::complex<double>>;

// =======================================================================================================
// PRINTING FUNCTIONS
// =======================================================================================================

void printCircuit(const Circuit& circuit) {
  for (const auto& instruction : circuit) {
    auto [tag, dag, dims, gate_type, target_qudits, params, control_set] =
        instruction;
    std::cout << "Tag: " << tag << "\n";
    std::cout << "Dag: " << dag << "\n";
    std::cout << "Dimensions: ";
    for (const auto& dim : dims) {
      std::cout << dim << " ";
    }
    std::cout << "\n";
    std::cout << "Gate Type: " << gate_type << "\n";
    std::cout << "Target Qudits: ";
    for (const auto& qubit : target_qudits) {
      std::cout << qubit << " ";
    }
    std::cout << "\n";

    // Printing control_set
    auto [control1, control2] = control_set;
    std::cout << "Control Set: ";
    for (const auto& control : control1) {
      std::cout << control << " ";
    }
    std::cout << "| ";
    for (const auto& control : control2) {
      std::cout << control << " ";
    }
    std::cout << "\n";
  }
}

// =======================================================================================================
// HELPER FUNCTIONS
// =======================================================================================================

bool isNoneOrEmpty(const py::object& obj) {
  if (obj.is_none())
    return true;

  if (py::isinstance<py::sequence>(obj)) {
    auto seq = obj.cast<py::sequence>();
    return seq.empty();
  }

  // Add additional checks for other types if needed

  return false;
}

bool checkDim(const std::vector<int>& dims,
              const std::variant<int, std::vector<int>>& target) {
  if (std::holds_alternative<int>(target)) {
    // If target is a single integer
    if (dims.size() != 1) {
      return false; // Different sizes, not exactly equal
    }
    return dims[0] == std::get<int>(target);
  }

  // If target is a vector
  const auto& targetVec = std::get<std::vector<int>>(target);
  if (dims.size() != targetVec.size()) {
    return false; // Different sizes, not exactly equal
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != targetVec[i]) {
      return false; // Different elements, not exactly equal
    }
  }
  return true; // All elements are the same, exactly equal
}

// Function to convert C++ vector of complex numbers to a Python list
py::list complexVectorToList(const CVec& vec) {
  py::list pyList;
  for (const auto& elem : vec) {
    try {
      // Convert std::complex<double> to Python complex object
      py::object pyComplex = py::cast(elem);
      // Append Python complex object to the list
      pyList.append(pyComplex);
    } catch (const std::exception& e) {
      // Handle any exceptions
      std::cerr << "Error appending Python object: " << e.what() << "\n";
    }
  }
  return pyList;
}

// =======================================================================================================
// PARSING FUNCTIONS
// =======================================================================================================

Circuit_info readCircuit(py::object& circ) {
  Circuit result;

  const auto numQudits = circ.attr("_num_qudits").cast<unsigned int>();
  const auto dimensions = circ.attr("_dimensions").cast<std::vector<size_t>>();

  // Get Python iterable
  py::iterator it = py::iter(circ.attr("instructions"));

  // Iterate over the Python iterable
  while (it != py::iterator::sentinel()) {
    const py::handle objHandle = *it;
    const auto obj = py::reinterpret_borrow<py::object>(objHandle);

    const auto tag = obj.attr("qasm_tag").cast<std::string>();

    const bool dagger = obj.attr("dagger").cast<bool>();

    const std::string gateType =
        py::cast<std::string>(obj.attr("gate_type").attr("name"));

    // Extracting dimensions
    const py::object dimsObj = obj.attr("_dimensions");
    std::vector<int> dims;
    if (py::isinstance<py::int_>(dimsObj)) {
      dims.push_back(py::cast<int>(dimsObj));
    } else if (py::isinstance<py::list>(dimsObj)) {
      dims = py::cast<std::vector<int>>(dimsObj);
    }

    // Extracting target_qudits
    const py::object targetQuditsObj = obj.attr("_target_qudits");
    std::vector<int> targetQudits;
    if (py::isinstance<py::int_>(targetQuditsObj)) {
      targetQudits.push_back(py::cast<int>(targetQuditsObj));
    } else if (py::isinstance<py::list>(targetQuditsObj)) {
      targetQudits = py::cast<std::vector<int>>(targetQuditsObj);
    }

    py::object params;
    if (isNoneOrEmpty(obj.attr("_params"))) {
    } else {
      params = obj.attr("_params");
    }

    std::tuple<std::vector<dd::QuantumRegister>, std::vector<dd::Control::Type>>
        controlSet = {};
    if (isNoneOrEmpty(obj.attr("_controls_data"))) {
      // std::cout << "control empty"<< "\n";
    } else {
      const py::object controlsData = obj.attr("_controls_data");
      auto indices =
          controlsData.attr("indices").cast<std::vector<dd::QuantumRegister>>();
      auto ctrlStates = controlsData.attr("ctrl_states")
                            .cast<std::vector<dd::Control::Type>>();

      controlSet = std::make_tuple(indices, ctrlStates);
    }

    result.emplace_back(tag, dagger, dims, gateType, targetQudits, params,
                        controlSet);

    // Increment the iterator
    ++it;
  }

  return std::make_tuple(numQudits, dimensions, result);
}

NoiseModel parseNoiseModel(const py::dict& noiseModel) {
  NoiseModel newNoiseModel;

  for (const auto& gate : noiseModel) {
    auto gateName = gate.first.cast<std::string>();

    const auto gateNoise = gate.second.cast<py::dict>();

    NoiseType newNoiseType;

    std::variant<std::string, std::vector<int>>
        noiseSpread; // Declared outside the if blocks

    for (const auto& noiseTypesPair : gateNoise) {
      if (py::isinstance<py::str>(noiseTypesPair.first)) {
        const auto noiseSpreadString = noiseTypesPair.first.cast<std::string>();
        noiseSpread = noiseSpreadString;
        // Handle string case
      } else if (py::isinstance<py::tuple>(noiseTypesPair.first)) {
        const auto noiseSpreadTuple =
            noiseTypesPair.first.cast<std::vector<int>>();
        noiseSpread = noiseSpreadTuple;
      }

      if (py::isinstance<py::dict>(noiseTypesPair.second)) {
        throw std::invalid_argument("Physical noise is not supported yet.");
      }

      const auto depo =
          noiseTypesPair.second.attr("probability_depolarizing").cast<double>();
      const auto deph =
          noiseTypesPair.second.attr("probability_dephasing").cast<double>();
      const std::tuple<double, double> noiseProb = std::make_tuple(depo, deph);

      newNoiseType[noiseSpread] = noiseProb;
    }
    newNoiseModel[gateName] = newNoiseType;
  }
  return newNoiseModel;
}

Circuit generateCircuit(const Circuit_info& circuitInfo,
                        const NoiseModel& noiseModel) {
  // Get current time in milliseconds
  auto currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  const auto& [num_qudits, dimensions, circuit] = circuitInfo;

  std::random_device rd;
  std::mt19937_64 gen(static_cast<uint64_t>(rd()) +
                      static_cast<uint64_t>(currentTimeMs));

  Circuit noisyCircuit;

  for (const Instruction& instruction : circuit) {
    noisyCircuit.push_back(instruction);

    const auto& [tag, dag, dims_gate, gate_type, target_qudits, params,
                 control_set] = instruction;
    std::vector<int> referenceLines(target_qudits.begin(), target_qudits.end());

    if (!(std::get<0>(control_set).empty()) &&
        (std::get<1>(control_set).empty())) {
      auto [ctrl_dits, levels] = control_set; // Decompose the tuple

      referenceLines.insert(referenceLines.end(), ctrl_dits.begin(),
                            ctrl_dits.end());
    }

    if (noiseModel.contains(tag)) {
      for (const auto& modeNoise : noiseModel.at(tag)) {
        auto mode = modeNoise.first;
        auto noiseInfo = modeNoise.second;
        const double depo = std::get<0>(noiseInfo);
        const double deph = std::get<1>(noiseInfo);

        std::discrete_distribution<int> xDist({1.0 - depo, depo});
        std::discrete_distribution<int> zDist({1.0 - deph, deph});

        const int xChoice = xDist(gen);
        const int zChoice = zDist(gen);

        if (xChoice == 1 || zChoice == 1) {
          std::vector<int> qudits;
          if (std::holds_alternative<std::vector<int>>(mode)) {
            qudits = std::get<std::vector<int>>(mode);

          } else if (std::holds_alternative<std::string>(mode)) {
            const std::string modeStr = std::get<std::string>(mode);

            if (modeStr == "local") {
              qudits = referenceLines;
            } else if (modeStr == "all") {
              for (int i = 0; std::cmp_less(i, num_qudits); ++i) {
                qudits.push_back(i);
              }
            } else if (modeStr == "nonlocal") {
              assert(gate_type == "TWO" || gate_type == "MULTI");
              qudits = referenceLines;
            } else if (modeStr == "control") {
              assert(gate_type == "TWO");
              qudits.push_back(target_qudits.at(0));
            } else if (modeStr == "target") {
              assert(gate_type == "TWO");
              qudits.push_back(target_qudits.at(1));
            }
          }
          if (xChoice == 1) {
            for (auto dit : qudits) {
              if (tag == "rxy" || tag == "rz" || tag == "virtrz") {
                std::vector<int> dims;
                dims.push_back(
                    static_cast<int>(dimensions[static_cast<uint64_t>(dit)]));
                py::list paramsNew;

                size_t value0 = 0;
                size_t value1 = 0;
                // Retrieve field 0 and 1 from params
                auto pl = params.cast<py::list>();
                value0 = pl[0].cast<size_t>();
                if (tag == "virtrz") {
                  if (dims.size() != 1) {
                    throw std::runtime_error(
                        "Dimension should be just an int"); // Different sizes,
                                                            // not exactly equal
                  }
                  if (std::cmp_not_equal(value0, dims[0] - 1)) {
                    value1 = value0 + 1;
                  } else {
                    value0 = static_cast<size_t>(dims[0] - 2);
                    value1 = static_cast<size_t>(dims[0] - 1);
                  }
                } else {
                  value1 = pl[1].cast<size_t>();
                }

                // Create a new list and append value0 and value1
                paramsNew.append(value0);
                paramsNew.append(value1);

                // Append pi and pi/2
                const auto pi = std::numbers::pi;
                const auto piOver2 = pi / 2.0;
                paramsNew.append(py::float_(pi));
                paramsNew.append(py::float_(piOver2));
                const Instruction newInst = std::make_tuple(
                    "rxy", false, dims, "SINGLE", std::vector<int>{dit},
                    py::cast<py::object>(paramsNew),
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(newInst);
              } else {
                const py::object paramsNew;
                std::vector<int> dims;
                dims.push_back(
                    static_cast<int>(dimensions[static_cast<uint64_t>(dit)]));

                const Instruction newInst = std::make_tuple(
                    "x", false, dims, "SINGLE", std::vector<int>{dit},
                    paramsNew,
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(newInst);
              }
            }
          }

          if (zChoice == 1) {
            for (auto dit : qudits) {
              if (tag == "rxy" || tag == "rz" || tag == "virtrz") {
                py::list paramsNew;

                std::vector<int> dims;
                dims.push_back(
                    static_cast<int>(dimensions[static_cast<uint64_t>(dit)]));

                size_t value0 = 0;
                size_t value1 = 0;
                // Retrieve field 0 and 1 from params
                auto pl = params.cast<py::list>();
                value0 = pl[0].cast<size_t>();
                if (tag == "virtrz") {
                  if (dims.size() != 1) {
                    throw std::runtime_error(
                        "Dimension should be just an int"); // Different
                                                            // sizes, not
                                                            // exactly equal
                  }
                  if (std::cmp_not_equal(value0, dims[0] - 1)) {
                    value1 = value0 + 1;
                  } else {
                    value0 = static_cast<size_t>(dims[0] - 2);
                    value1 = static_cast<size_t>(dims[0] - 1);
                  }
                } else {
                  value1 = pl[1].cast<size_t>();
                }

                // Create a new list and append value0 and value1
                paramsNew.append(value0);
                paramsNew.append(value1);

                // Append pi and pi/2
                const auto pi = std::numbers::pi;
                paramsNew.append(py::float_(pi));
                const Instruction newInst = std::make_tuple(
                    "rz", false, dims, "SINGLE", std::vector<int>{dit},
                    py::cast<py::object>(paramsNew),
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(newInst);
              } else {
                const py::object paramsNew;
                std::vector<int> dims;
                dims.push_back(
                    static_cast<int>(dimensions[static_cast<uint64_t>(dit)]));

                const Instruction newInst = std::make_tuple(
                    "z", false, dims, "SINGLE", std::vector<int>{dit},
                    paramsNew,
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(newInst);
              }
            }
          }
        }
      }
    }
  }

  return noisyCircuit;
}

// =======================================================================================================
// SIMULATION FUNCTIONS
// =======================================================================================================

/*
 * SUPPORTED GATES AT THE MOMENT UNTIL DIMENSION 7
"csum": "csum",
"cx": "cx",
"h": "h",
"rxy": "r",
"rz": "rz",
"rh": "rh",
"virtrz": "virtrz",
"s": "s",
"x": "x",
"z": "z"
 */
using ddpkg = std::unique_ptr<dd::MDDPackage>;

dd::MDDPackage::mEdge getGate(const ddpkg& dd, const Instruction& instruction) {
  const auto& [tag, dag, dims, gate_type, target_qudits, params, control_set] =
      instruction;

  dd::MDDPackage::mEdge gate;
  auto numberRegs =
      static_cast<dd::QuantumRegisterCount>(dd->numberOfQuantumRegisters);

  dd::QuantumRegister tq = 0;
  tq = static_cast<dd::QuantumRegister>(target_qudits.at(0));

  const dd::Controls controlSet{};
  if ((!std::get<0>(control_set).empty()) &&
      (!std::get<1>(control_set).empty())) {
    const std::vector<dd::QuantumRegister> ctrlQudits =
        std::get<0>(control_set);
    const std::vector<dd::Control::Type> ctrlLevels = std::get<1>(control_set);
  }

  if (tag == "rxy") {
    // Handle rxy tag with dimension 2
    auto pl = params.cast<py::list>();
    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();
    auto theta = pl[2].cast<double>();
    auto phi = pl[3].cast<double>();

    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::RXY(theta, phi);
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      // Handle rxy tag with dimension 3
      const dd::TritMatrix matrix = dd::RXY3(theta, phi, leva, levb);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      // Handle rxy tag with dimension 4
      const dd::QuartMatrix matrix = dd::RXY4(theta, phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::RXY5(theta, phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::RXY6(theta, phi, leva, levb);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::RXY7(theta, phi, leva, levb);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "rz") {
    auto pl = params.cast<py::list>();

    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();
    auto phi = pl[2].cast<double>();

    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::RZ(phi);
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::RZ3(phi, leva, levb);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::RZ4(phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::RZ5(phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::RZ6(phi, leva, levb);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::RZ7(phi, leva, levb);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "rh") {
    auto pl = params.cast<py::list>();
    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();

    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::RH();
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::RH3(leva, levb);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::RH4(leva, levb);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::RH5(leva, levb);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::RH6(leva, levb);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::RH7(leva, levb);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "virtrz") {
    auto pl = params.cast<py::list>();
    auto leva = pl[0].cast<size_t>();
    auto phi = pl[1].cast<double>();

    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::VirtRZ(phi, leva);
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::VirtRZ3(phi, leva);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::VirtRZ4(phi, leva);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::VirtRZ5(phi, leva);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::VirtRZ6(phi, leva);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::VirtRZ7(phi, leva);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "x") {
    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::Xmat;
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::X3;
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::X4;
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::X5;
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::X6;
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::X7;
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "s") {
    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::Smat;
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::S3();
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::S4();
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::S5();
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::S6();
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::S7();
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "z") {
    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::Zmat;
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::Z3();
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::Z4();
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::Z5();
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::Z6();
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::Z7();
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }

  } else if (tag == "h") {
    if (checkDim(dims, 2)) {
      const dd::GateMatrix matrix = dd::H();
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      const dd::TritMatrix matrix = dd::H3();
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      const dd::QuartMatrix matrix = dd::H4();
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      const dd::QuintMatrix matrix = dd::H5();
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      const dd::SextMatrix matrix = dd::H6();
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      const dd::SeptMatrix matrix = dd::H7();
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "cx") {
    auto pl = params.cast<py::list>();

    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();
    auto ctrlLev = pl[2].cast<dd::Control::Type>();
    auto phi = pl[3].cast<dd::fp>();

    auto cReg = static_cast<dd::QuantumRegister>(target_qudits.at(0));
    auto target = static_cast<dd::QuantumRegister>(target_qudits.at(1));
    return dd->cex(numberRegs, ctrlLev, phi, leva, levb, cReg, target, dag);
  } else if (tag == "csum") {
    auto cReg = static_cast<dd::QuantumRegister>(target_qudits.at(0));
    auto target = static_cast<dd::QuantumRegister>(target_qudits.at(1));
    return dd->csum(numberRegs, cReg, target, dag);
  }
  if (dag) {
    gate = dd->conjugateTranspose(gate);
  }
  return gate;
}

CVec ddsimulator(dd::QuantumRegisterCount numLines,
                 const std::vector<size_t>& dims, const Circuit& circuit) {
  const ddpkg dd = std::make_unique<dd::MDDPackage>(numLines, dims);
  auto psi = dd->makeZeroState(numLines);

  for (const Instruction& instruction : circuit) {
    dd::MDDPackage::mEdge gate;
    try {
      gate = getGate(dd, instruction);
    } catch (const std::exception& e) {
      std::cerr << "Caught exception in gate creation: " << e.what() << "\n";
      throw; // Re-throw the exception to propagate it further
    }
    try {
      psi = dd->multiply(gate, psi);
    } catch (const std::exception& e) {
      printCircuit(circuit);
      std::cout << "THE MATRIX\n";
      dd->getVectorizedMatrix(gate);
      std::cout << "THE VECTOR\n";
      dd->printVector(psi);
      std::cerr << "Problem is in multiplication " << e.what() << "\n";
      throw; // Re-throw the exception to propagate it further
    }
  }
  return dd->getVector(psi);
}

py::list stateVectorSimulation(py::object& circ, py::object& noiseModel) {
  auto parsedCircuitInfo = readCircuit(circ);
  auto [numQudits, dims, original_circuit] = parsedCircuitInfo;

  Circuit noisyCircuit = original_circuit;
  const auto noiseModelDict =
      noiseModel.attr("quantum_errors").cast<py::dict>();
  const NoiseModel newNoiseModel = parseNoiseModel(noiseModelDict);
  noisyCircuit = generateCircuit(parsedCircuitInfo, newNoiseModel);

  const CVec myList =
      ddsimulator(static_cast<dd::QuantumRegisterCount>(numQudits),
                  static_cast<std::vector<size_t>>(dims), noisyCircuit);

  py::list result = complexVectorToList(myList);

  return result;
}

} // namespace

// NOLINTNEXTLINE(misc-include-cleaner)
PYBIND11_MODULE(MQT_QUDITS_MODULE_NAME, m, py::mod_gil_not_used()) {
  auto misim = m.def_submodule("misim");
  misim.def("state_vector_simulation", &stateVectorSimulation, "circuit"_a,
            "noise_model"_a);
}
