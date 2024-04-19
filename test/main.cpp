#include "dd/MDDPackage.hpp"

#include <memory>

int main() { // NOLINT(bugprone-exception-escape)
  std::vector<std::size_t> const lines{2, 3};
  dd::QuantumRegisterCount numLines = 2U;

  auto dd = std::make_unique<dd::MDDPackage>(
      numLines, lines); // Create new package instance capable of handling a
                        // qubit and a qutrit
  auto zeroState = dd->makeZeroState(numLines); // zero_state = |0>

  /* Creating a DD requires the following inputs:
   * 1. A matrix describing a single-qubit/qudit operation (here: the Hadamard
   * matrix)
   * 2. The number of qudits the DD will operate on (here: two lines)
   * 3. The operations are applied to the qubit q0 and the qutrit q1
   * (4. Controlled operations can be created by additionally specifying a list
   * of control qubits before the target declaration)
   */
  auto hOnQubit = dd->makeGateDD<dd::GateMatrix>(dd::H(), numLines, 0);
  // auto h_on_qutrit = dd->makeGateDD<dd::TritMatrix>(dd::H3(), 2, 1);

  // Multiplying the operation and the state results in a new state, here a
  // single qubit in superposition
  auto psi = dd->multiply(hOnQubit, zeroState);

  // Multiplying the operation and the state results in a new state, here a
  // single qutrit in superposition psi = dd->multiply(h_on_qutrit, zero_state);

  // An example of how to create a set of controls and add them together to
  // create a more complex controlled operation
  dd::Controls control{};
  const dd::Control c{0, 1};
  control.insert(c);

  // An example of a controlled qutrit X operation, controlled on the level 1 of
  // the qubit
  auto cex = dd->makeGateDD<dd::TritMatrix>(dd::X3, numLines, control, 1);

  psi = dd->multiply(cex, psi);

  // The last lines retrieves the state vector and prints it
  dd->printVector(psi);
}
