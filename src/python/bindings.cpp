#include "dd/MDDPackage.hpp"

#include <any>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

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
// =======================================================================================================
// =======================================================================================================
// ============@@@@@@@@@@@@@@@@@@@@ PRINTING FUNCTION
// @@@@@@@@@@@@@@@@@@@@================================
// =======================================================================================================
// =======================================================================================================

// Function to print NoiseType
void printNoiseType(const NoiseType& noiseType, int indent = 0) {
  for (const auto& [key, noise] : noiseType) {
    // Print the key
    for (int i = 0; i < indent; ++i)
      std::cout << "    ";
    if (std::holds_alternative<std::string>(key))
      std::cout << std::get<std::string>(key) << " -> ";
    else {
      std::cout << "[";
      for (int val : std::get<std::vector<int>>(key))
        std::cout << val << " ";
      std::cout << "] -> ";
    }

    // Print the noise tuple
    std::cout << "(" << std::get<0>(noise) << ", " << std::get<1>(noise) << ")"
              << std::endl;
  }
}

// Function to print NoiseModel
void printNoiseModel(const NoiseModel& noiseModel) {
  for (const auto& [key, noiseType] : noiseModel) {
    std::cout << key << ":" << std::endl;
    printNoiseType(noiseType, 1);
  }
}

void printCvec(CVec vector) {
  for (const auto& cn : vector) {
    std::cout << cn << std::endl;
  }
}

void printCircuit(const Circuit& circuit) {
  for (const auto& instruction : circuit) {
    auto [tag, dag, dims, gate_type, target_qudits, params, control_set] =
        instruction;
    std::cout << "Tag: " << tag << std::endl;
    std::cout << "Dag: " << dag << std::endl;
    std::cout << "Dimensions: ";
    for (const auto& dim : dims) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "Gate Type: " << gate_type << std::endl;
    std::cout << "Target Qudits: ";
    for (const auto& qubit : target_qudits) {
      std::cout << qubit << " ";
    }
    std::cout << std::endl;

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
    std::cout << std::endl;
  }
}

// =======================================================================================================
// =======================================================================================================
// =======================================================================================================
// ============================================= HELPER FUNCTION
// =========================================
// =======================================================================================================
// =======================================================================================================

bool is_none_or_empty(const py::object& obj) {
  if (obj.is_none())
    return true;

  if (py::isinstance<py::sequence>(obj)) {
    auto seq = obj.cast<py::sequence>();
    return seq.size() == 0;
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
  } else {
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
}

// Function to convert C++ vector of complex numbers to a Python list
py::list complex_vector_to_list(const CVec& vec) {
  py::list pyList;
  for (const auto& elem : vec) {
    try {
      // Convert std::complex<double> to Python complex object
      py::object pyComplex = py::cast(elem);
      // Append Python complex object to the list
      pyList.append(pyComplex);
    } catch (const std::exception& e) {
      // Handle any exceptions
      std::cerr << "Error appending Python object: " << e.what() << std::endl;
    }
  }
  return pyList;
}

// =======================================================================================================
// =======================================================================================================
// =======================================================================================================
// ===============::::::::::::::::::::::PARSING FUNCTIONS
// ::::::::::::::::::::::===========================
// =======================================================================================================
// =======================================================================================================

Circuit_info readCircuit(py::object& circ) {
  Circuit result;

  unsigned int num_qudits = circ.attr("_num_qudits").cast<unsigned int>();
  std::vector<size_t> dimensions =
      circ.attr("_dimensions").cast<std::vector<size_t>>();

  bool result_empty = py::isinstance<py::none>(circ.attr("instructions"));

  // Get Python iterable
  py::iterator it = py::iter(circ.attr("instructions"));

  // Iterate over the Python iterable
  while (it != py::iterator::sentinel()) {
    py::handle obj_handle = *it;
    py::object obj = py::reinterpret_borrow<py::object>(obj_handle);

    std::string tag = obj.attr("qasm_tag").cast<std::string>();

    bool dagger = obj.attr("dagger").cast<bool>();

    std::string gate_type =
        py::cast<std::string>(obj.attr("gate_type").attr("name"));

    // Extracting dimensions
    py::object dims_obj = obj.attr("_dimensions");
    std::vector<int> dims;
    if (py::isinstance<py::int_>(dims_obj)) {
      dims.push_back(py::cast<int>(dims_obj));
    } else if (py::isinstance<py::list>(dims_obj)) {
      dims = py::cast<std::vector<int>>(dims_obj);
    }

    // Extracting target_qudits
    py::object target_qudits_obj = obj.attr("_target_qudits");
    std::vector<int> target_qudits;
    if (py::isinstance<py::int_>(target_qudits_obj)) {
      target_qudits.push_back(py::cast<int>(target_qudits_obj));
    } else if (py::isinstance<py::list>(target_qudits_obj)) {
      target_qudits = py::cast<std::vector<int>>(target_qudits_obj);
    }

    py::object params;
    if (is_none_or_empty(obj.attr("_params"))) {
    } else {
      params = obj.attr("_params");
    }

    std::tuple<std::vector<dd::QuantumRegister>, std::vector<dd::Control::Type>>
        control_set = {};
    if (is_none_or_empty(obj.attr("_controls_data"))) {
      // std::cout << "control empty"<< std::endl;
    } else {
      py::object controls_data = obj.attr("_controls_data");
      auto indices = controls_data.attr("indices")
                         .cast<std::vector<dd::QuantumRegister>>();
      auto ctrlStates = controls_data.attr("ctrl_states")
                            .cast<std::vector<dd::Control::Type>>();

      control_set = std::make_tuple(indices, ctrlStates);
    }

    result.push_back(std::make_tuple(tag, dagger, dims, gate_type,
                                     target_qudits, params, control_set));

    // Increment the iterator
    ++it;
  }

  return std::make_tuple(num_qudits, dimensions, result);
}

NoiseModel parse_noise_model(const py::dict& noise_model) {
  NoiseModel newNoiseModel;

  for (const auto& gate : noise_model) {
    auto gateName = gate.first.cast<std::string>();

    py::dict gateNoise = gate.second.cast<py::dict>();

    NoiseType newNoiseType;

    std::variant<std::string, std::vector<int>>
        noiseSpread; // Declared outside the if blocks

    for (const auto& noiseTypesPair : gateNoise) {
      if (py::isinstance<py::str>(noiseTypesPair.first)) {
        std::string noiseSpreadString =
            noiseTypesPair.first.cast<std::string>();
        noiseSpread = noiseSpreadString;
        // Handle string case
      } else if (py::isinstance<py::tuple>(noiseTypesPair.first)) {
        std::vector<int> noiseSpreadTuple =
            noiseTypesPair.first.cast<std::vector<int>>();
        noiseSpread = noiseSpreadTuple;
      }

      double depo =
          noiseTypesPair.second.attr("probability_depolarizing").cast<double>();
      double deph =
          noiseTypesPair.second.attr("probability_dephasing").cast<double>();
      std::tuple<double, double> noiseProb = std::make_tuple(depo, deph);

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
  std::mt19937_64 gen(rd() + currentTimeMs);

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

    if (noiseModel.find(tag) != noiseModel.end()) {
      for (const auto& mode_noise : noiseModel.at(tag)) {
        auto mode = mode_noise.first;
        auto noise_info = mode_noise.second;
        double depo = std::get<0>(noise_info);
        double deph = std::get<1>(noise_info);

        std::discrete_distribution<int> x_dist({1.0 - depo, depo});
        std::discrete_distribution<int> z_dist({1.0 - deph, deph});

        int x_choice = x_dist(gen);
        int z_choice = z_dist(gen);

        if (x_choice == 1 || z_choice == 1) {
          std::vector<int> qudits;
          if (std::holds_alternative<std::vector<int>>(mode)) {
            qudits = std::get<std::vector<int>>(mode);

          } else if (std::holds_alternative<std::string>(mode)) {
            std::string modeStr = std::get<std::string>(mode);

            if (modeStr == "local") {
              qudits = referenceLines;
            } else if (modeStr == "all") {
              for (int i = 0; i < num_qudits; ++i)
                qudits.push_back(i);
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
          if (x_choice == 1) {
            for (auto dit : qudits) {
              if (tag == "rxy" || tag == "rz" || tag == "virtrz") {
                std::vector<int> dims;
                dims.push_back(static_cast<int>(
                    dimensions[static_cast<unsigned long>(dit)]));
                py::list params_new;

                size_t value_0, value_1;
                // Retrieve field 0 and 1 from params
                auto pl = params.cast<py::list>();
                value_0 = pl[0].cast<size_t>();
                if (tag == "virtrz") {
                  if (dims.size() != 1) {
                    throw std::runtime_error(
                        "Dimension should be just an int"); // Different sizes,
                                                            // not exactly equal
                  }
                  if (value_0 != dims[0] - 1) {
                    value_1 = value_0 + 1;
                  } else {
                    value_0 = dims[0] - 2;
                    value_1 = dims[0] - 1;
                  }
                } else {
                  value_1 = pl[1].cast<size_t>();
                }

                // Create a new list and append value_0 and value_1
                params_new.append(value_0);
                params_new.append(value_1);

                // Append pi and pi/2
                double pi = 3.14159265358979323846;
                double pi_over_2 = pi / 2.0;
                params_new.append(py::float_(pi));
                params_new.append(py::float_(pi_over_2));
                Instruction new_inst = std::make_tuple(
                    "rxy", false, dims, "SINGLE", std::vector<int>{dit},
                    py::cast<py::object>(params_new),
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(new_inst);
              } else {
                py::object params_new;
                std::vector<int> dims;
                dims.push_back(static_cast<int>(
                    dimensions[static_cast<unsigned long>(dit)]));

                Instruction new_inst = std::make_tuple(
                    "x", false, dims, "SINGLE", std::vector<int>{dit},
                    params_new,
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(new_inst);
              }
            }
          }

          if (z_choice == 1) {
            for (auto dit : qudits) {
              if (tag == "rxy" || tag == "rz" || tag == "virtrz") {
                py::list params_new;

                std::vector<int> dims;
                dims.push_back(static_cast<int>(
                    dimensions[static_cast<unsigned long>(dit)]));

                size_t value_0, value_1;
                // Retrieve field 0 and 1 from params
                auto pl = params.cast<py::list>();
                value_0 = pl[0].cast<size_t>();
                if (tag == "virtrz") {
                  if (dims.size() != 1) {
                    throw std::runtime_error(
                        "Dimension should be just an int"); // Different sizes,
                                                            // not exactly equal
                  }
                  if (value_0 != dims[0] - 1) {
                    value_1 = value_0 + 1;
                  } else {
                    value_0 = dims[0] - 2;
                    value_1 = dims[0] - 1;
                  }
                } else {
                  value_1 = pl[1].cast<size_t>();
                }

                // Create a new list and append value_0 and value_1
                params_new.append(value_0);
                params_new.append(value_1);

                // Append pi and pi/2
                double pi = 3.14159265358979323846;
                params_new.append(py::float_(pi));
                Instruction newInst = std::make_tuple(
                    "rz", false, dims, "SINGLE", std::vector<int>{dit},
                    py::cast<py::object>(params_new),
                    std::tuple<std::vector<dd::QuantumRegister>,
                               std::vector<dd::Control::Type>>());
                noisyCircuit.push_back(newInst);
              } else {
                py::object paramsNew;
                std::vector<int> dims;
                dims.push_back(static_cast<int>(
                    dimensions[static_cast<unsigned long>(dit)]));

                Instruction newInst = std::make_tuple(
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
// =======================================================================================================
// =======================================================================================================
// ===============()()()()()()()()()()() SIMULATION FUNCTIONS
// ()()()()()()()()()==========================
// =======================================================================================================
// =======================================================================================================
// =======================================================================================================
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

  dd::Controls controlSet{};
  if ((std::get<0>(control_set).size() > 0) &&
      (std::get<1>(control_set).size() > 0)) {
    std::vector<dd::QuantumRegister> ctrlQudits = std::get<0>(control_set);
    std::vector<dd::Control::Type> ctrlLevels = std::get<1>(control_set);
  }

  if (tag == "rxy") {
    // Handle rxy tag with dimension 2
    auto pl = params.cast<py::list>();
    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();
    auto theta = pl[2].cast<double>();
    auto phi = pl[3].cast<double>();

    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::RXY(theta, phi);
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      // Handle rxy tag with dimension 3
      dd::TritMatrix matrix = dd::RXY3(theta, phi, leva, levb);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      // Handle rxy tag with dimension 4
      dd::QuartMatrix matrix = dd::RXY4(theta, phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::RXY5(theta, phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::RXY6(theta, phi, leva, levb);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::RXY7(theta, phi, leva, levb);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "rz") {
    auto pl = params.cast<py::list>();

    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();
    auto phi = pl[2].cast<double>();

    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::RZ(phi);
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::RZ3(phi, leva, levb);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::RZ4(phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::RZ5(phi, leva, levb);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::RZ6(phi, leva, levb);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::RZ7(phi, leva, levb);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "rh") {
    auto pl = params.cast<py::list>();
    auto leva = pl[0].cast<size_t>();
    auto levb = pl[1].cast<size_t>();

    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::RH();
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::RH3(leva, levb);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::RH4(leva, levb);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::RH5(leva, levb);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::RH6(leva, levb);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::RH7(leva, levb);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "virtrz") {
    auto pl = params.cast<py::list>();
    auto leva = pl[0].cast<size_t>();
    auto phi = pl[1].cast<double>();

    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::VirtRZ(phi, leva);
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::VirtRZ3(phi, leva);
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::VirtRZ4(phi, leva);
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::VirtRZ5(phi, leva);
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::VirtRZ6(phi, leva);
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::VirtRZ7(phi, leva);
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "x") {
    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::Xmat;
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::X3;
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::X4;
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::X5;
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::X6;
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::X7;
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "s") {
    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::Smat;
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::S3();
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::S4();
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::S5();
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::S6();
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::S7();
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }
  } else if (tag == "z") {
    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::Zmat;
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::Z3();
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::Z4();
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::Z5();
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::Z6();
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::Z7();
      gate = dd->makeGateDD<dd::SeptMatrix>(matrix, numberRegs, controlSet, tq);
    }

  } else if (tag == "h") {
    if (checkDim(dims, 2)) {
      dd::GateMatrix matrix = dd::H();
      gate = dd->makeGateDD<dd::GateMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 3)) {
      dd::TritMatrix matrix = dd::H3();
      gate = dd->makeGateDD<dd::TritMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 4)) {
      dd::QuartMatrix matrix = dd::H4();
      gate =
          dd->makeGateDD<dd::QuartMatrix>(matrix, numberRegs, controlSet, tq);

    } else if (checkDim(dims, 5)) {
      dd::QuintMatrix matrix = dd::H5();
      gate =
          dd->makeGateDD<dd::QuintMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 6)) {
      dd::SextMatrix matrix = dd::H6();
      gate = dd->makeGateDD<dd::SextMatrix>(matrix, numberRegs, controlSet, tq);
    } else if (checkDim(dims, 7)) {
      dd::SeptMatrix matrix = dd::H7();
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
      std::cerr << "Caught exception in gate creation: " << e.what()
                << std::endl;
      throw; // Re-throw the exception to propagate it further
    }
    try {
      psi = dd->multiply(gate, psi);
    } catch (const std::exception& e) {
      printCircuit(circuit);
      std::cout << "THE MATRIX  " << std::endl;
      dd->getVectorizedMatrix(gate);
      std::cout << "THE VECTOR  " << std::endl;
      dd->printVector(psi);
      std::cerr << "Problem is in multiplication " << e.what() << std::endl;
      throw; // Re-throw the exception to propagate it further
    }
  }
  return dd->getVector(psi);
}

py::list stateVectorSimulation(py::object& circ, py::object& noiseModel) {
  auto parsedCircuitInfo = readCircuit(circ);
  auto [numQudits, dims, original_circuit] = parsedCircuitInfo;

  Circuit noisyCircuit = original_circuit;
  py::dict noiseModelDict = noiseModel.attr("quantum_errors").cast<py::dict>();
  NoiseModel newNoiseModel = parse_noise_model(noiseModelDict);
  noisyCircuit = generateCircuit(parsedCircuitInfo, newNoiseModel);

  CVec myList =
      ddsimulator(static_cast<dd::QuantumRegisterCount>(numQudits),
                  static_cast<std::vector<size_t>>(dims), noisyCircuit);

  py::list result = complex_vector_to_list(myList);

  return result;
}

PYBIND11_MODULE(_qudits, m) {
  auto misim = m.def_submodule("misim");
  misim.def("state_vector_simulation", &stateVectorSimulation, "circuit"_a,
            "noise_model"_a);
}
