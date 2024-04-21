/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */
#ifndef DDMDDPackage_H
#define DDMDDPackage_H

#include "Complex.hpp"
#include "ComplexNumbers.hpp"
#include "ComplexTable.hpp"
#include "ComplexValue.hpp"
#include "ComputeTable.hpp"
#include "Control.hpp"
#include "Definitions.hpp"
#include "Edge.hpp"
#include "GateMatrixDefinitions.hpp"
#include "UnaryComputeTable.hpp"
#include "UniqueTable.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dd {
class MDDPackage {
  ///
  /// Complex number handling
  ///
public:
  ComplexNumbers complexNumber{};

  ///
  /// Construction, destruction, information and reset
  ///
public:
  static constexpr std::size_t MAX_POSSIBLE_REGISTERS =
      static_cast<std::make_unsigned_t<QuantumRegister>>(
          std::numeric_limits<QuantumRegister>::max()) +
      1U;
  static constexpr std::size_t DEFAULT_REGISTERS = 128;

  explicit MDDPackage(std::size_t nqr, std::vector<size_t> sizes)
      : numberOfQuantumRegisters(nqr), registersSizes(std::move(sizes)) {
    resize(nqr);
  };

  ~MDDPackage() = default;

  MDDPackage(const MDDPackage& MDDPackage) = delete; // no copy constructor
  MDDPackage& operator=(const MDDPackage& MDDPackage) =
      delete; // no copy assignment constructor

  // TODO RESIZE
  //  resize the package instance
  void resize(std::size_t nq) {
    // TODO DISCUSS THIS FEATURE
    if (nq > MAX_POSSIBLE_REGISTERS) {
      throw std::invalid_argument(
          "Requested too many qubits from package. Qubit datatype only "
          "allows up to " +
          std::to_string(MAX_POSSIBLE_REGISTERS) + " qubits, while " +
          std::to_string(nq) +
          " were requested. Please recompile the package with a wider "
          "Qubit type!");
    }
    numberOfQuantumRegisters = nq;
    vUniqueTable.resize(numberOfQuantumRegisters);
    mUniqueTable.resize(numberOfQuantumRegisters);
    // dUniqueTable.resize(number_of_quantum_registers);
    // stochasticNoiseOperationCache.resize(number_of_quantum_registers);
    idTable.resize(numberOfQuantumRegisters);
  }

  // reset package state
  void reset() {
    // TODO IMPLEMENT
    // clearUniqueTables();
    // clearComputeTables();
    complexNumber.clear();
  }

  // TODO CHECK SYNTAX OF SETTERS AND GETTERS
  //  getter for number qudits

  [[nodiscard]] auto qregisters() const { return numberOfQuantumRegisters; }

  // setter dimensionalisties
  [[nodiscard]] auto registerDimensions(const std::vector<size_t>& regs) {
    registersSizes = regs;
  }

  // getter for sizes
  [[nodiscard]] auto regsSize() const { return registersSizes; }

public:
  std::size_t numberOfQuantumRegisters;
  // TODO THIS IS NOT CONST RIGHT?
  // from LSB TO MSB
  std::vector<size_t> registersSizes;

  ///
  /// Vector nodes, edges and quantum states
  ///
public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  struct vNode {
    std::vector<Edge<vNode>> edges{}; // edges out of this node
    vNode* next{};                    // used to link nodes in unique table
    RefCount refCount{}; // reference count, how many active dd are using
                         // the node
    QuantumRegister
        varIndx{}; // variable index (nonterminal) value (-1 for terminal),
                   // index in the circuit endianness 0 from below

    static vNode
        terminalNode; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    constexpr static vNode* terminal{
        &terminalNode}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)

    static constexpr bool isTerminal(const vNode* nodePoint) {
      return nodePoint == terminal;
    }
  };
  using vEdge = Edge<vNode>;
  using vCachedEdge = CachedEdge<vNode>;

  vEdge normalize(const vEdge& edge, bool cached) {
    std::vector<bool> zero;
    // find indices that are not zero
    std::vector<std::size_t> nonZeroIndices;
    std::size_t counter = 0UL;
    for (auto const& i : edge.nextNode->edges) {
      if (i.weight.approximatelyZero()) {
        zero.push_back(true);
      } else {
        zero.push_back(false);
        nonZeroIndices.push_back(counter);
      }
      counter++;
    }

    // make sure to release cached numbers approximately zero, but not exactly
    // zero
    if (cached) {
      for (auto i = 0UL; i < zero.size(); i++) {
        if (zero.at(i) && edge.nextNode->edges.at(i).weight != Complex::zero) {
          complexNumber.returnToCache(edge.nextNode->edges.at(i).weight);
          edge.nextNode->edges.at(i) = vEdge::zero;
        }
      }
    }

    // all equal to zero
    if (none_of(cbegin(zero), cend(zero), std::logical_not<>())) {
      if (!cached && !edge.isTerminal()) {
        // If it is not a cached computation, the node has to be put back into
        // the chain
        vUniqueTable.returnNode(edge.nextNode);
      }
      return vEdge::zero;
    }

    if (nonZeroIndices.size() == 1) {
      // search for first element different from zero
      auto currentEdge = edge;
      auto& weightFromChild =
          currentEdge.nextNode->edges.at(nonZeroIndices.front()).weight;

      if (cached && weightFromChild != Complex::one) {
        currentEdge.weight = weightFromChild;
      } else {
        currentEdge.weight = complexNumber.lookup(weightFromChild);
      }

      weightFromChild = Complex::one;
      return currentEdge;
    }

    // calculate normalizing factor
    auto sumNorm2 = ComplexNumbers::mag2(edge.nextNode->edges.at(0).weight);
    auto mag2Max = ComplexNumbers::mag2(edge.nextNode->edges.at(0).weight);
    auto argMax = 0UL;

    // TODO FIX BECAUSE AT THIS STAGE IT TRIES ALWAYS TO GET THE FIRST EDGE AND
    // I WANT THE FIRST BEH BASED ON PREVIOUS CODE
    for (auto i = 1UL; i < edge.nextNode->edges.size(); i++) {
      sumNorm2 =
          sumNorm2 + ComplexNumbers::mag2(edge.nextNode->edges.at(i).weight);
    }
    for (auto i = 1UL; i <= edge.nextNode->edges.size(); i++) {
      auto counterBack = edge.nextNode->edges.size() - i;
      if (ComplexNumbers::mag2(edge.nextNode->edges.at(counterBack).weight) +
              ComplexTable<>::tolerance() >=
          mag2Max) {
        mag2Max =
            ComplexNumbers::mag2(edge.nextNode->edges.at(counterBack).weight);
        argMax = counterBack;
      }
    }

    const auto norm = std::sqrt(sumNorm2);
    const auto magMax = std::sqrt(mag2Max);
    const auto commonFactor = norm / magMax;

    // set incoming edge weight to max
    auto currentEdge = edge;
    auto& max = currentEdge.nextNode->edges.at(argMax);

    if (cached && max.weight != Complex::one) {
      // if(cached && !currentEdge.weight.approximatelyOne()){
      currentEdge.weight = max.weight;
      currentEdge.weight.real->value *= commonFactor;
      currentEdge.weight.img->value *= commonFactor;
    } else {
      auto realPart = CTEntry::val(currentEdge.weight.real) * commonFactor;
      auto imgPart = CTEntry::val(currentEdge.weight.img) * commonFactor;
      currentEdge.weight = complexNumber.lookup(realPart, imgPart);
      if (currentEdge.weight.approximatelyZero()) {
        return vEdge::zero;
      }
    }

    max.weight = complexNumber.lookup(magMax / norm, 0.);
    if (max.weight == Complex::zero) {
      max = vEdge::zero;
    }

    // actual normalization of the edges
    // TODO CHECK IF CHANGE MADE IN THE CACHED IF IS CORRECT
    for (auto i = 0UL; i < edge.nextNode->edges.size(); ++i) {
      if (i != argMax) {
        auto& iEdge = edge.nextNode->edges.at(i);

        if (cached &&
            iEdge.weight != Complex::zero) { // TODO CHECK EXACTLY HERE
          complexNumber.returnToCache(iEdge.weight);
          ComplexNumbers::div(iEdge.weight, iEdge.weight, currentEdge.weight);
          iEdge.weight = complexNumber.lookup(iEdge.weight);
        } else {
          auto c = complexNumber.getTemporary();
          ComplexNumbers::div(c, iEdge.weight, currentEdge.weight);
          iEdge.weight = complexNumber.lookup(c);
        }
        if (iEdge.weight == Complex::zero) {
          iEdge = vEdge::zero;
        }
      }
    }

    return currentEdge;
  }

  // generate |0...0> with N quantum registers
  vEdge makeZeroState(QuantumRegisterCount n, std::size_t start = 0) {
    if (n + start > numberOfQuantumRegisters) {
      // TODO UNDERSTAND RESIZING
      throw std::runtime_error("Requested state with " +
                               std::to_string(n + start) +
                               " QUANTUM REGISTERS, but current package "
                               "configuration only supports up to " +
                               std::to_string(numberOfQuantumRegisters) +
                               " QUANTUM REGISTERS. Please allocate a "
                               "larger package instance.");
    }
    auto first = vEdge::one;
    for (std::size_t nodeIdx = start; nodeIdx < n + start; nodeIdx++) {
      std::vector<Edge<vNode>> newOutgoingEdges;
      newOutgoingEdges.reserve(registersSizes.at(nodeIdx));
      newOutgoingEdges.push_back(first);
      for (auto i = 1U; i < registersSizes.at(nodeIdx); i++) {
        newOutgoingEdges.push_back(vEdge::zero);
      }

      first =
          makeDDNode(static_cast<QuantumRegister>(nodeIdx), newOutgoingEdges);
    }
    return first;
  }

  // generate computational basis state |i> with n quantum registers
  vEdge makeBasisState(QuantumRegisterCount n, const std::vector<size_t>& state,
                       std::size_t start = 0) {
    if (n + start > numberOfQuantumRegisters) {
      throw std::runtime_error(
          "Requested state with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up "
          "to " +
          std::to_string(numberOfQuantumRegisters) +
          " qubits. Please allocate a larger package instance.");
    }
    auto f = vEdge::one;
    for (std::size_t pos = start; pos < n + start; ++pos) {
      std::vector<vEdge> edges(registersSizes.at(pos), vEdge::zero);
      edges.at(state.at(pos)) = f;
      f = makeDDNode(static_cast<QuantumRegister>(pos), edges);
    }
    return f;
  }

  // create a normalized DD node and return an edge pointing to it. The
  // node is not recreated if it already exists.
  template <class Node>
  Edge<Node> makeDDNode(QuantumRegister varidx,
                        const std::vector<Edge<Node>>& edges,
                        bool cached = false) {
    auto& uniqueTable = getUniqueTable<Node>();

    Edge<Node> newEdge{uniqueTable.getNode(), Complex::one};
    newEdge.nextNode->varIndx = varidx;
    newEdge.nextNode->edges = edges;

    assert(newEdge.nextNode->refCount == 0);

    for ([[maybe_unused]] const auto& edge : edges) {
      assert(edge.nextNode->varIndx == varidx - 1 || edge.isTerminal());
    }

    // normalize it
    newEdge = normalize(newEdge, cached);
    assert(newEdge.nextNode->varIndx == varidx || newEdge.isTerminal());

    // look it up in the unique tables
    auto lookedUpEdge = uniqueTable.lookup(newEdge, false);
    assert(lookedUpEdge.nextNode->varIndx == varidx ||
           lookedUpEdge.isTerminal());

    // set specific node properties for matrices
    if constexpr (std::is_same_v<Node, mNode>) {
      if (lookedUpEdge.nextNode == newEdge.nextNode) {
        checkSpecialMatrices(lookedUpEdge.nextNode);
      }
    }

    return lookedUpEdge;
  }

public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  struct mNode {
    std::vector<Edge<mNode>> edges{}; // edges out of this node
    mNode* next{};                    // used to link nodes in unique table
    RefCount refCount{};              // reference count
    QuantumRegister varIndx{};        // variable index (nonterminal) value (-1
                                      // for terminal)
    bool symmetric = false;           // node is symmetric
    bool identity = false;            // node resembles identity

    static mNode
        terminalNode; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    constexpr static mNode* terminal{
        &terminalNode}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)

    static constexpr bool isTerminal(const mNode* nodePoint) {
      return nodePoint == terminal;
    }
  };
  using mEdge = Edge<mNode>;
  using mCachedEdge = CachedEdge<mNode>;

  mEdge normalize(const mEdge& edge, bool cached) {
    auto argmax = -1;

    std::vector<bool> zero;
    zero.reserve(edge.nextNode->edges.size());
    for (auto& i : edge.nextNode->edges) {
      zero.emplace_back(i.weight.approximatelyZero());
    }

    // make sure to release cached numbers approximately zero, but not
    // exactly zero
    if (cached) {
      for (auto i = 0U; i < zero.size(); i++) {
        if (zero.at(i) && edge.nextNode->edges.at(i).weight != Complex::zero) {
          // TODO what is returnToCache

          complexNumber.returnToCache(edge.nextNode->edges.at(i).weight);
          edge.nextNode->edges.at(i) = mEdge::zero;
        }
      }
    }

    fp maxMagnitude = 0;
    auto maxWeight = Complex::one;
    // determine max amplitude
    for (auto i = 0U; i < zero.size(); ++i) {
      if (zero.at(i)) {
        continue;
      }
      if (argmax == -1) {
        argmax = static_cast<decltype(argmax)>(i);
        maxMagnitude = ComplexNumbers::mag2(edge.nextNode->edges.at(i).weight);
        maxWeight = edge.nextNode->edges.at(i).weight;
      } else {
        auto currentMagnitude =
            ComplexNumbers::mag2(edge.nextNode->edges.at(i).weight);
        if (currentMagnitude - maxMagnitude > ComplexTable<>::tolerance()) {
          argmax = static_cast<decltype(argmax)>(i);
          maxMagnitude = currentMagnitude;
          maxWeight = edge.nextNode->edges.at(i).weight;
        }
      }
    }

    // all equal to zero
    if (argmax == -1) {
      if (!cached && !edge.isTerminal()) {
        // If it is not a cached computation, the node has to be put
        // back into the chain
        mUniqueTable.returnNode(edge.nextNode);
      }
      return mEdge::zero;
    }

    auto currentEdge = edge;
    // divide each entry by max
    for (auto i = 0U; i < edge.nextNode->edges.size(); ++i) {
      if (static_cast<decltype(argmax)>(i) == argmax) {
        if (cached) {
          if (currentEdge.weight == Complex::one) {
            currentEdge.weight = maxWeight;
          } else {
            ComplexNumbers::mul(currentEdge.weight, currentEdge.weight,
                                maxWeight);
          }
        } else {
          if (currentEdge.weight == Complex::one) {
            currentEdge.weight = maxWeight;
          } else {
            auto newComplexNumb = complexNumber.getTemporary();
            ComplexNumbers::mul(newComplexNumb, currentEdge.weight, maxWeight);
            currentEdge.weight = complexNumber.lookup(newComplexNumb);
          }
        }
        currentEdge.nextNode->edges.at(i).weight = Complex::one;
      } else {
        if (cached && !zero.at(i) &&
            currentEdge.nextNode->edges.at(i).weight != Complex::one) {
          complexNumber.returnToCache(currentEdge.nextNode->edges.at(i).weight);
        }
        if (currentEdge.nextNode->edges.at(i).weight.approximatelyOne()) {
          currentEdge.nextNode->edges.at(i).weight = Complex::one;
        }
        auto newComplexNumb = complexNumber.getTemporary();

        ComplexNumbers::div(newComplexNumb,
                            currentEdge.nextNode->edges.at(i).weight,
                            maxWeight);
        currentEdge.nextNode->edges.at(i).weight =
            complexNumber.lookup(newComplexNumb);
      }
    }
    return currentEdge;
  }

  /// Make GATE DD
  // SIZE => EDGE (number of successors)
  // build matrix representation for a single gate on an n-qubit circuit

  template <typename Matrix>
  mEdge makeGateDD(const Matrix& mat, QuantumRegisterCount n,
                   QuantumRegister target, std::size_t start = 0) {
    return makeGateDD(mat, n, Controls{}, target, start);
  }

  template <typename Matrix>
  mEdge makeGateDD(const Matrix& mat, QuantumRegisterCount n,
                   const Control& control, QuantumRegister target,
                   std::size_t start = 0) {
    return makeGateDD(mat, n, Controls{control}, target, start);
  }

  template <typename Matrix>
  mEdge makeGateDD(const Matrix& mat, QuantumRegisterCount n,
                   const Controls& controls, QuantumRegister target,
                   std::size_t start = 0) {
    if (n + start > numberOfQuantumRegisters) {
      throw std::runtime_error(
          "Requested gate with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up "
          "to " +
          std::to_string(numberOfQuantumRegisters) +
          " qubits. Please allocate a larger package instance.");
    }

    auto targetRadix = registersSizes.at(static_cast<std::size_t>(target));
    auto edges = targetRadix * targetRadix;
    std::vector<mEdge> edgesMat(edges, mEdge::zero);

    auto currentControl = controls.begin();

    for (auto i = 0U; i < edges; ++i) {
      if (mat.at(i).r != 0 || mat.at(i).i != 0) {
        edgesMat.at(i) = mEdge::terminal(complexNumber.lookup(mat.at(i)));
      }
    }
    auto currentReg = static_cast<QuantumRegister>(start);
    // process lines below target
    for (; currentReg < target; currentReg++) {
      auto radix = registersSizes.at(static_cast<std::size_t>(currentReg));
      for (auto rowMat = 0U; rowMat < targetRadix; ++rowMat) {
        for (auto colMat = 0U; colMat < targetRadix; ++colMat) {
          auto entryPos = (rowMat * targetRadix) + colMat;

          std::vector<mEdge> quadEdges(radix * radix, mEdge::zero);

          if (currentControl != controls.end() &&
              currentControl->quantumRegister == currentReg) {
            if (rowMat == colMat) {
              for (auto i = 0U; i < radix; i++) {
                auto diagInd = i * radix + i;
                if (i == currentControl->type) {
                  quadEdges.at(diagInd) = edgesMat.at(entryPos);
                } else {
                  quadEdges.at(diagInd) =
                      makeIdent(static_cast<QuantumRegister>(start),
                                static_cast<QuantumRegister>(currentReg - 1));
                }
              }
            } else {
              quadEdges.at(currentControl->type +
                           radix * currentControl->type) =
                  edgesMat.at(entryPos);
            }
            edgesMat.at(entryPos) = makeDDNode(currentReg, quadEdges);

          } else { // not connected
            for (auto iD = 0U; iD < radix; iD++) {
              quadEdges.at(iD * radix + iD) = edgesMat.at(entryPos);
            }
            edgesMat.at(entryPos) = makeDDNode(currentReg, quadEdges);
          }
        }
      }

      if (currentControl != controls.end() &&
          currentControl->quantumRegister == currentReg) {
        ++currentControl;
      }
    }

    // target line
    auto targetNodeEdge = makeDDNode(currentReg, edgesMat);

    // process lines above target
    for (; currentReg < static_cast<QuantumRegister>(n - 1 + start);
         currentReg++) {
      auto nextReg = static_cast<QuantumRegister>(currentReg + 1);
      auto nextRadix = registersSizes.at(static_cast<std::size_t>(nextReg));
      std::vector<mEdge> nextEdges(nextRadix * nextRadix, mEdge::zero);

      if (currentControl != controls.end() &&
          currentControl->quantumRegister == nextReg) {
        for (auto i = 0U; i < nextRadix; i++) {
          auto diagInd = i * nextRadix + i;
          if (i == currentControl->type) {
            nextEdges.at(diagInd) = targetNodeEdge;
          } else {
            nextEdges.at(diagInd) =
                makeIdent(static_cast<QuantumRegister>(start),
                          static_cast<QuantumRegister>(nextReg - 1));
          }
        }

        ++currentControl;

      } else { // not connected
        for (auto iD = 0U; iD < nextRadix; iD++) {
          nextEdges.at(iD * nextRadix + iD) = targetNodeEdge;
        }
      }
      targetNodeEdge = makeDDNode(nextReg, nextEdges);
    }
    return targetNodeEdge;
  }

  ///
  /// Identity matrices
  ///
public:
  // create n-qudit identity DD. makeIdent(n) === makeIdent(0, n-1)
  mEdge makeIdent(QuantumRegisterCount n) {
    return makeIdent(0, static_cast<QuantumRegister>(n - 1));
  }

  mEdge makeIdent(QuantumRegister leastSignificantQubit,
                  QuantumRegister mostSignificantQubit) {
    if (mostSignificantQubit < leastSignificantQubit) {
      return mEdge::one;
    }

    if (leastSignificantQubit == 0 &&
        idTable.at(static_cast<std::size_t>(mostSignificantQubit)).nextNode !=
            nullptr) {
      return idTable.at(static_cast<std::size_t>(mostSignificantQubit));
    }

    if (mostSignificantQubit >= 1 &&
        (idTable.at(static_cast<std::size_t>(mostSignificantQubit) - 1))
                .nextNode != nullptr) {
      auto basicDimMost =
          registersSizes.at(static_cast<std::size_t>(mostSignificantQubit));
      std::vector<mEdge> identityEdges{};

      for (auto i = 0UL; i < basicDimMost; i++) {
        for (auto j = 0UL; j < basicDimMost; j++) {
          if (i == j) {
            identityEdges.push_back(
                idTable[static_cast<std::size_t>(mostSignificantQubit) - 1]);
          } else {
            identityEdges.push_back(mEdge::zero);
          }
        }
      }
      idTable.at(static_cast<std::size_t>(mostSignificantQubit)) = makeDDNode(
          static_cast<QuantumRegister>(mostSignificantQubit), identityEdges);

      return idTable.at(static_cast<std::size_t>(mostSignificantQubit));
    }

    // create an Identity DD from scratch
    auto basicDimLeast =
        registersSizes.at(static_cast<std::size_t>(leastSignificantQubit));
    std::vector<mEdge> identityEdgesLeast{};

    for (auto i = 0UL; i < basicDimLeast; i++) {
      for (auto j = 0UL; j < basicDimLeast; j++) {
        if (i == j) {
          identityEdgesLeast.push_back(mEdge::one);
        } else {
          identityEdgesLeast.push_back(mEdge::zero);
        }
      }
    }

    auto e = makeDDNode(static_cast<QuantumRegister>(leastSignificantQubit),
                        identityEdgesLeast);

    for (std::size_t intermediaryRegs =
             static_cast<std::size_t>(leastSignificantQubit) + 1;
         intermediaryRegs <= static_cast<std::size_t>(mostSignificantQubit);
         intermediaryRegs++) {
      auto basicDimInt = registersSizes.at(intermediaryRegs);
      std::vector<mEdge> identityEdgesInt{};

      for (auto i = 0UL; i < basicDimInt; i++) {
        for (auto j = 0UL; j < basicDimInt; j++) {
          if (i == j) {
            identityEdgesInt.push_back(e);
          } else {
            identityEdgesInt.push_back(mEdge::zero);
          }
        }
      }
      e = makeDDNode(static_cast<QuantumRegister>(intermediaryRegs),
                     identityEdgesInt);
    }

    if (leastSignificantQubit == 0) {
      idTable.at(static_cast<std::size_t>(mostSignificantQubit)) = e;
    }
    return e;
  }

  // identity table access and reset
  [[nodiscard]] const auto& getIdentityTable() const { return idTable; }

  void clearIdentityTable() {
    for (auto& entry : idTable) {
      entry.nextNode = nullptr;
    }
  }

private:
  std::vector<mEdge> idTable{};

public:
  ///
  /// Addition
  ///
  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge> vectorAdd{};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge> matrixAdd{};

  template <class Node>
  [[nodiscard]] ComputeTable<CachedEdge<Node>, CachedEdge<Node>,
                             CachedEdge<Node>>&
  getAddComputeTable();

  template <class Edge> Edge add(const Edge& x, const Edge& y) {
    [[maybe_unused]] const auto before = complexNumber.cacheCount();

    auto result = add2(x, y);

    if (result.weight != Complex::zero) {
      complexNumber.returnToCache(result.weight);
      result.weight = complexNumber.lookup(result.weight);
    }

    [[maybe_unused]] const auto after = complexNumber.complexCache.getCount();
    assert(after == before);

    return result;
  }

  template <class Node>
  Edge<Node> add2(const Edge<Node>& x, const Edge<Node>& y) {
    // no sum performed
    if (x.nextNode == nullptr) {
      return y;
    }
    if (y.nextNode == nullptr) {
      return x;
    }

    if (x.weight == Complex::zero) {
      if (y.weight == Complex::zero) {
        return y;
      }
      auto result = y;
      result.weight = complexNumber.getCached(CTEntry::val(y.weight.real),
                                              CTEntry::val(y.weight.img));
      return result;
    }
    if (y.weight == Complex::zero) {
      auto result = x;
      result.weight = complexNumber.getCached(CTEntry::val(x.weight.real),
                                              CTEntry::val(x.weight.img));
      return result;
    }
    if (x.nextNode == y.nextNode) {
      auto result = y;
      result.weight = complexNumber.addCached(x.weight, y.weight);
      if (result.weight.approximatelyZero()) {
        complexNumber.returnToCache(result.weight);
        return Edge<Node>::zero;
      }
      return result;
    }

    auto& computeTable = getAddComputeTable<Node>();
    auto result =
        computeTable.lookup({x.nextNode, x.weight}, {y.nextNode, y.weight});
    if (result.nextNode != nullptr) {
      if (result.weight.approximatelyZero()) {
        return Edge<Node>::zero;
      }
      return {result.nextNode, complexNumber.getCached(result.weight)};
    }

    QuantumRegister newSuccessor = 0;

    if (x.isTerminal()) {
      newSuccessor = y.nextNode->varIndx;
    } else {
      newSuccessor = x.nextNode->varIndx;
      if (!y.isTerminal() && y.nextNode->varIndx > newSuccessor) {
        newSuccessor = y.nextNode->varIndx;
      }
    }

    // constexpr std::size_t     N = std::tuple_size_v<decltype(x.p->e)>;
    // TODO CHECK HERE IF MAKES SENSE
    std::vector<Edge<Node>> edgeSum(x.nextNode->edges.size(),
                                    dd::Edge<Node>::zero);

    for (auto i = 0U; i < x.nextNode->edges.size(); i++) {
      Edge<Node> e1{};

      if (!x.isTerminal() && x.nextNode->varIndx == newSuccessor) {
        e1 = x.nextNode->edges.at(i);

        if (e1.weight != Complex::zero) {
          e1.weight = complexNumber.mulCached(e1.weight, x.weight);
        }
      } else {
        e1 = x;
        if (y.nextNode->edges.at(i).nextNode == nullptr) {
          e1 = {nullptr, Complex::zero};
        }
      }

      Edge<Node> e2{};
      if (!y.isTerminal() && y.nextNode->varIndx == newSuccessor) {
        e2 = y.nextNode->edges.at(i);

        if (e2.weight != Complex::zero) {
          e2.weight = complexNumber.mulCached(e2.weight, y.weight);
        }
      } else {
        e2 = y;
        if (x.nextNode->edges.at(i).nextNode == nullptr) {
          e2 = {nullptr, Complex::zero};
        }
      }

      edgeSum.at(i) = add2(e1, e2);

      if (!x.isTerminal() && x.nextNode->varIndx == newSuccessor &&
          e1.weight != Complex::zero) {
        complexNumber.returnToCache(e1.weight);
      }

      if (!y.isTerminal() && y.nextNode->varIndx == newSuccessor &&
          e2.weight != Complex::zero) {
        complexNumber.returnToCache(e2.weight);
      }
    }

    auto e = makeDDNode(newSuccessor, edgeSum, true);
    computeTable.insert({x.nextNode, x.weight}, {y.nextNode, y.weight},
                        {e.nextNode, e.weight});
    return e;
  }
  ///
  /// Multiplication
  ///
public:
  ComputeTable<mEdge, vEdge, vCachedEdge> matrixVectorMultiplication{};
  ComputeTable<mEdge, mEdge, mCachedEdge> matrixMatrixMultiplication{};

  template <class LeftOperandNode, class RightOperandNode>
  [[nodiscard]] ComputeTable<Edge<LeftOperandNode>, Edge<RightOperandNode>,
                             CachedEdge<RightOperandNode>>&
  getMultiplicationComputeTable();

  template <class LeftOperand, class RightOperand>
  RightOperand multiply(const LeftOperand& x, const RightOperand& y,
                        dd::QuantumRegister start = 0) {
    [[maybe_unused]] const auto before = complexNumber.cacheCount();

    QuantumRegister var = -1;

    if (!x.isTerminal()) {
      var = x.nextNode->varIndx;
    }
    if (!y.isTerminal() && (y.nextNode->varIndx) > var) {
      var = y.nextNode->varIndx;
    }

    RightOperand e = multiply2(x, y, var, start);

    if (e.weight != Complex::zero && e.weight != Complex::one) {
      complexNumber.returnToCache(e.weight);
      e.weight = complexNumber.lookup(e.weight);
    }

    [[maybe_unused]] const auto after = complexNumber.cacheCount();
    assert(before == after);

    return e;
  }

private:
  template <class LeftOperandNode, class RightOperandNode>
  Edge<RightOperandNode>
  multiply2(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
            QuantumRegister var, QuantumRegister start = 0) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    using ResultEdge = Edge<RightOperandNode>;

    if (x.nextNode == nullptr) {
      return {nullptr, Complex::zero};
    }
    if (y.nextNode == nullptr) {
      return y;
    }

    if (x.weight == Complex::zero || y.weight == Complex::zero) {
      return ResultEdge::zero;
    }

    if (var == start - 1) {
      return ResultEdge::terminal(complexNumber.mulCached(x.weight, y.weight));
    }

    auto xCopy = x;
    xCopy.weight = Complex::one;
    auto yCopy = y;
    yCopy.weight = Complex::one;

    auto& computeTable =
        getMultiplicationComputeTable<LeftOperandNode, RightOperandNode>();
    auto lookupResult = computeTable.lookup(xCopy, yCopy);

    if (lookupResult.nextNode != nullptr) {
      if (lookupResult.weight.approximatelyZero()) {
        return ResultEdge::zero;
      }

      auto resEdgeInit = ResultEdge{
          lookupResult.nextNode, complexNumber.getCached(lookupResult.weight)};

      ComplexNumbers::mul(resEdgeInit.weight, resEdgeInit.weight, x.weight);
      ComplexNumbers::mul(resEdgeInit.weight, resEdgeInit.weight, y.weight);

      if (resEdgeInit.weight.approximatelyZero()) {
        complexNumber.returnToCache(resEdgeInit.weight);
        return ResultEdge::zero;
      }
      return resEdgeInit;
    }

    ResultEdge resultEdge{};

    if (x.nextNode->varIndx == var &&
        x.nextNode->varIndx == y.nextNode->varIndx) {
      if (x.nextNode->identity) {
        if constexpr (std::is_same_v<RightOperandNode, mNode>) {
          // additionally check if y is the identity in case of matrix
          // multiplication
          if (y.nextNode->identity) {
            resultEdge = makeIdent(start, var);
          } else {
            resultEdge = yCopy;
          }
        } else {
          resultEdge = yCopy;
        }

        computeTable.insert(xCopy, yCopy,
                            {resultEdge.nextNode, resultEdge.weight});
        resultEdge.weight = complexNumber.mulCached(x.weight, y.weight);

        if (resultEdge.weight.approximatelyZero()) {
          complexNumber.returnToCache(resultEdge.weight);
          return ResultEdge::zero;
        }
        return resultEdge;
      }

      if constexpr (std::is_same_v<RightOperandNode, mNode>) {
        // additionally check if y is the identity in case of matrix
        // multiplication
        if (y.nextNode->identity) {
          resultEdge = xCopy;
          computeTable.insert(xCopy, yCopy,
                              {resultEdge.nextNode, resultEdge.weight});
          resultEdge.weight = complexNumber.mulCached(x.weight, y.weight);

          if (resultEdge.weight.approximatelyZero()) {
            complexNumber.returnToCache(resultEdge.weight);
            return ResultEdge::zero;
          }
          return resultEdge;
        }
      }
    }

    // TODO CHECK AGAIN THIS COULD BE WRONG
    const std::size_t rows =
        x.isTerminal()
            ? 1U
            : registersSizes.at(static_cast<std::size_t>(x.nextNode->varIndx));
    const std::size_t cols =
        (std::is_same_v<RightOperandNode, mNode>)
            ? y.isTerminal() ? 1U
                             : registersSizes.at(static_cast<std::size_t>(
                                   y.nextNode->varIndx))
            : 1U;
    const std::size_t multiplicationBoundary =
        x.isTerminal()
            ? (y.isTerminal() ? 1U
                              : registersSizes.at(static_cast<std::size_t>(
                                    y.nextNode->varIndx)))
            : registersSizes.at(static_cast<std::size_t>(x.nextNode->varIndx));

    std::vector<ResultEdge> edge(multiplicationBoundary * cols,
                                 ResultEdge::zero);

    for (auto i = 0U; i < rows; i++) {
      for (auto j = 0U; j < cols; j++) {
        auto idx = cols * i + j;
        // edge.at(idx) = ResultEdge::zero;

        for (auto k = 0U; k < multiplicationBoundary; k++) {
          LEdge e1{};
          if (!x.isTerminal() && x.nextNode->varIndx == var) {
            e1 = x.nextNode->edges.at(rows * i + k);
          } else {
            e1 = xCopy;
          }

          REdge e2{};
          if (!y.isTerminal() && y.nextNode->varIndx == var) {
            e2 = y.nextNode->edges.at(j + cols * k);
          } else {
            e2 = yCopy;
          }

          auto multipliedRecurRes =
              multiply2(e1, e2, static_cast<QuantumRegister>(var - 1), start);

          if (k == 0 || edge.at(idx).weight == Complex::zero) {
            edge.at(idx) = multipliedRecurRes;
          } else if (multipliedRecurRes.weight != Complex::zero) {
            auto oldEdge = edge.at(idx);
            edge.at(idx) = add2(edge.at(idx), multipliedRecurRes);
            complexNumber.returnToCache(oldEdge.weight);
            complexNumber.returnToCache(multipliedRecurRes.weight);
          }
        }
      }
    }
    resultEdge = makeDDNode(var, edge, true);

    computeTable.insert(xCopy, yCopy, {resultEdge.nextNode, resultEdge.weight});

    if (resultEdge.weight != Complex::zero &&
        (x.weight != Complex::one || y.weight != Complex::one)) {
      if (resultEdge.weight == Complex::one) {
        resultEdge.weight = complexNumber.mulCached(x.weight, y.weight);
      } else {
        ComplexNumbers::mul(resultEdge.weight, resultEdge.weight, x.weight);
        ComplexNumbers::mul(resultEdge.weight, resultEdge.weight, y.weight);
      }
      if (resultEdge.weight.approximatelyZero()) {
        complexNumber.returnToCache(resultEdge.weight);
        return ResultEdge::zero;
      }
    }
    return resultEdge;
  }
  ///
  /// Inner product, fidelity, expectation value
  ///
public:
  ComputeTable<vEdge, vEdge, vCachedEdge> vectorInnerProduct{};

  ComplexValue innerProduct(const vEdge& x, const vEdge& y) {
    if (x.nextNode == nullptr || y.nextNode == nullptr ||
        x.weight.approximatelyZero() ||
        y.weight.approximatelyZero()) { // the 0 case
      return {0, 0};
    }

    [[maybe_unused]] const auto before = complexNumber.cacheCount();

    auto circWidth = x.nextNode->varIndx;
    if (y.nextNode->varIndx > circWidth) {
      circWidth = y.nextNode->varIndx;
    }
    const ComplexValue ip =
        innerProduct(x, y, static_cast<QuantumRegister>(circWidth + 1));

    [[maybe_unused]] const auto after = complexNumber.cacheCount();
    assert(after == before);

    return ip;
  }

  fp fidelity(const vEdge& x, const vEdge& y) {
    const auto fid = innerProduct(x, y);
    return fid.r * fid.r + fid.i * fid.i;
  }

private:
  ComplexValue innerProduct(const vEdge& x, const vEdge& y,
                            QuantumRegister var) {
    if (x.nextNode == nullptr || y.nextNode == nullptr ||
        x.weight.approximatelyZero() ||
        y.weight.approximatelyZero()) { // the 0 case
      return {0.0, 0.0};
    }

    if (var == 0) {
      auto c = complexNumber.getTemporary();
      ComplexNumbers::mul(c, x.weight, y.weight);
      return {c.real->value, c.img->value};
    }

    auto xCopy = x;
    xCopy.weight = Complex::one;
    auto yCopy = y;
    yCopy.weight = Complex::one;

    auto nodeLookup = vectorInnerProduct.lookup(xCopy, yCopy);
    if (nodeLookup.nextNode != nullptr) {
      auto c = complexNumber.getTemporary(nodeLookup.weight);
      ComplexNumbers::mul(c, c, x.weight);
      ComplexNumbers::mul(c, c, y.weight);
      return {CTEntry::val(c.real), CTEntry::val(c.img)};
    }

    auto width = static_cast<QuantumRegister>(var - 1);

    ComplexValue sum{0.0, 0.0};
    for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(width));
         i++) {
      vEdge e1{};
      if (!x.isTerminal() && x.nextNode->varIndx == width) {
        e1 = x.nextNode->edges.at(i);
      } else {
        e1 = xCopy;
      }
      vEdge e2{};
      if (!y.isTerminal() && y.nextNode->varIndx == width) {
        e2 = y.nextNode->edges.at(i);
        e2.weight = ComplexNumbers::conj(e2.weight);
      } else {
        e2 = yCopy;
      }
      auto cv = innerProduct(e1, e2, width);
      sum.r += cv.r;
      sum.i += cv.i;
    }
    nodeLookup.nextNode = vNode::terminal;
    nodeLookup.weight = sum;

    vectorInnerProduct.insert(xCopy, yCopy, nodeLookup);
    auto c = complexNumber.getTemporary(sum);
    ComplexNumbers::mul(c, c, x.weight);
    ComplexNumbers::mul(c, c, y.weight);
    return {CTEntry::val(c.real), CTEntry::val(c.img)};
  }

  ///
  /// Kronecker/tensor product
  ///
public:
  ComputeTable<vEdge, vEdge, vCachedEdge, 4096> vectorKronecker{};
  ComputeTable<mEdge, mEdge, mCachedEdge, 4096> matrixKronecker{};

  template <class Node>
  [[nodiscard]] ComputeTable<Edge<Node>, Edge<Node>, CachedEdge<Node>, 4096>&
  getKroneckerComputeTable();

  template <class Edge>
  Edge kronecker(const Edge& x, const Edge& y, bool incIdx = true) {
    auto e = kronecker2(x, y, incIdx);

    if (e.weight != Complex::zero && e.weight != Complex::one) {
      complexNumber.returnToCache(e.weight);
      e.weight = complexNumber.lookup(e.weight);
    }

    return e;
  }

  // extent the DD pointed to by `e` with `h` identities on top and `l`
  // identities at the bottom
  mEdge extend(const mEdge& e, QuantumRegister h, QuantumRegister l = 0) {
    auto f = (l > 0)
                 ? kronecker(e, makeIdent(static_cast<QuantumRegisterCount>(l)))
                 : e;
    auto g = (h > 0)
                 ? kronecker(makeIdent(static_cast<QuantumRegisterCount>(h)), f)
                 : f;
    return g;
  }

private:
  template <class Node>
  Edge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y,
                        bool incIdx = true) {
    if (x.weight.approximatelyZero() || y.weight.approximatelyZero()) {
      return Edge<Node>::zero;
    }

    if (x.isTerminal()) {
      auto r = y;
      r.weight = complexNumber.mulCached(x.weight, y.weight);
      return r;
    }

    auto& computeTable = getKroneckerComputeTable<Node>();
    auto r = computeTable.lookup(x, y);
    if (r.nextNode != nullptr) {
      if (r.weight.approximatelyZero()) {
        return Edge<Node>::zero;
      }
      return {r.nextNode, complexNumber.getCached(r.weight)};
    }

    if (x.nextNode->identity) {
      std::vector<Edge<Node>> newEdges(x.nextNode->edges.size(),
                                       dd::Edge<Node>::zero);

      for (auto i = 0U;
           i < registersSizes.at(static_cast<std::size_t>(x.nextNode->varIndx));
           i++) {
        newEdges.at(i + i * (registersSizes.at(static_cast<std::size_t>(
                                x.nextNode->varIndx)))) = y;
      }
      auto idx = incIdx ? static_cast<QuantumRegister>(y.nextNode->varIndx + 1)
                        : y.nextNode->varIndx;

      auto e = makeDDNode(idx, newEdges);

      for (auto i = 0; i < x.nextNode->varIndx; ++i) {
        std::vector<Edge<Node>> eSucc(e.nextNode->edges.size(),
                                      dd::Edge<Node>::zero);
        for (auto j = 0U;
             j <
             registersSizes.at(static_cast<std::size_t>(e.nextNode->varIndx));
             j++) {
          eSucc.at(j + j * (registersSizes.at(static_cast<std::size_t>(
                               e.nextNode->varIndx)))) = e;
        }

        idx = incIdx ? static_cast<QuantumRegister>(e.nextNode->varIndx + 1)
                     : e.nextNode->varIndx;

        e = makeDDNode(idx, eSucc);
      }

      e.weight = complexNumber.getCached(CTEntry::val(y.weight.real),
                                         CTEntry::val(y.weight.img));
      computeTable.insert(x, y, {e.nextNode, e.weight});
      return e;
    }

    std::vector<Edge<Node>> edge(x.nextNode->edges.size(),
                                 dd::Edge<Node>::zero);
    for (auto i = 0U; i < x.nextNode->edges.size(); ++i) {
      edge.at(i) = kronecker2(x.nextNode->edges.at(i), y, incIdx);
    }

    auto idx = incIdx ? static_cast<QuantumRegister>(y.nextNode->varIndx +
                                                     x.nextNode->varIndx + 1)
                      : x.nextNode->varIndx;
    auto e = makeDDNode(idx, edge, true);
    ComplexNumbers::mul(e.weight, e.weight, x.weight);
    computeTable.insert(x, y, {e.nextNode, e.weight});
    return e;
  }

public:
  mEdge cex(QuantumRegisterCount numberRegs, dd::Control::Type level, fp phi,
            size_t leva, size_t levb, QuantumRegister cReg,
            QuantumRegister target, bool isDagger = false) {
    const dd::Control control{cReg, level};

    if (registersSizes.at(static_cast<std::size_t>(target)) == 2) {
      const auto matrix = dd::embX2(phi);
      auto gate =
          makeGateDD<dd::GateMatrix>(matrix, numberRegs, control, target);
      if (isDagger) {
        gate = conjugateTranspose(gate);
      }
      return gate;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 3) {
      const auto matrix = dd::embX3(phi, leva, levb);
      auto gate =
          makeGateDD<dd::TritMatrix>(matrix, numberRegs, control, target);
      if (isDagger) {
        gate = conjugateTranspose(gate);
      }
      return gate;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 4) {
      const auto matrix = dd::embX4(phi, leva, levb);
      auto gate =
          makeGateDD<dd::QuartMatrix>(matrix, numberRegs, control, target);
      if (isDagger) {
        gate = conjugateTranspose(gate);
      }
      return gate;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 5) {
      const auto matrix = dd::embX5(phi, leva, levb);
      auto gate =
          makeGateDD<dd::QuintMatrix>(matrix, numberRegs, control, target);
      if (isDagger) {
        gate = conjugateTranspose(gate);
      }
      return gate;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 6) {
      const auto matrix = dd::embX6(phi, leva, levb);
      auto gate =
          makeGateDD<dd::SextMatrix>(matrix, numberRegs, control, target);
      if (isDagger) {
        gate = conjugateTranspose(gate);
      }
      return gate;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 7) {
      const auto matrix = dd::embX7(phi, leva, levb);
      auto gate =
          makeGateDD<dd::SeptMatrix>(matrix, numberRegs, control, target);
      if (isDagger) {
        gate = conjugateTranspose(gate);
      }
      return gate;
    }
    throw std::invalid_argument("Dimensions of target not implemented");
  }

  mEdge csum(QuantumRegisterCount numberRegs, QuantumRegister cReg,
             QuantumRegister target, bool isDagger = false) {
    auto res = makeIdent(numberRegs);
    if (registersSizes.at(static_cast<std::size_t>(target)) == 2) {
      for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(cReg));
           i++) {
        const dd::Control control{cReg, static_cast<dd::Control::Type>(i)};
        const auto matrix = dd::Xmat;
        auto gate =
            makeGateDD<dd::GateMatrix>(matrix, numberRegs, control, target);
        for (auto counter = 0U; counter < i; counter++) {
          res = multiply(res, gate);
        }
      }
      if (isDagger) {
        res = conjugateTranspose(res);
      }
      return res;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 3) {
      for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(cReg));
           i++) {
        const dd::Control control{cReg, static_cast<dd::Control::Type>(i)};
        const auto matrix = dd::X3;
        auto gate =
            makeGateDD<dd::TritMatrix>(matrix, numberRegs, control, target);
        for (auto counter = 0U; counter < i; counter++) {
          res = multiply(res, gate);
        }
      }
      if (isDagger) {
        res = conjugateTranspose(res);
      }
      return res;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 4) {
      for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(cReg));
           i++) {
        const dd::Control control{cReg, static_cast<dd::Control::Type>(i)};
        const auto matrix = dd::X4;
        auto gate =
            makeGateDD<dd::QuartMatrix>(matrix, numberRegs, control, target);
        for (auto counter = 0U; counter < i; counter++) {
          res = multiply(res, gate);
        }
      }
      if (isDagger) {
        res = conjugateTranspose(res);
      }
      return res;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 5) {
      for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(cReg));
           i++) {
        const dd::Control control{cReg, static_cast<dd::Control::Type>(i)};
        const auto matrix = dd::X5;
        auto gate =
            makeGateDD<dd::QuintMatrix>(matrix, numberRegs, control, target);
        for (auto counter = 0U; counter < i; counter++) {
          res = multiply(res, gate);
        }
      }
      if (isDagger) {
        res = conjugateTranspose(res);
      }
      return res;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 6) {
      for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(cReg));
           i++) {
        const dd::Control control{cReg, static_cast<dd::Control::Type>(i)};
        const auto matrix = dd::X6;
        auto gate =
            makeGateDD<dd::SextMatrix>(matrix, numberRegs, control, target);
        for (auto counter = 0U; counter < i; counter++) {
          res = multiply(res, gate);
        }
      }
      if (isDagger) {
        res = conjugateTranspose(res);
      }
      return res;
    }
    if (registersSizes.at(static_cast<std::size_t>(target)) == 7) {
      for (auto i = 0U; i < registersSizes.at(static_cast<std::size_t>(cReg));
           i++) {
        const dd::Control control{cReg, static_cast<dd::Control::Type>(i)};
        const auto matrix = dd::X7;
        auto gate =
            makeGateDD<dd::SeptMatrix>(matrix, numberRegs, control, target);
        for (auto counter = 0U; counter < i; counter++) {
          res = multiply(res, gate);
        }
      }
      if (isDagger) {
        res = conjugateTranspose(res);
      }
      return res;
    }
    throw std::invalid_argument("Dimensions of target not implemented");
  }

  vEdge spread2(QuantumRegisterCount n,
                const std::vector<QuantumRegister>& lines, vEdge& state) {
    dd::Controls const control01{{lines.at(0), 1}};
    auto cH = makeGateDD<dd::GateMatrix>(dd::Hmat, n, control01, lines.at(1));

    dd::Controls const control10{{lines.at(1), 0}};
    mEdge minus = mEdge::zero;
    mEdge xp10 = mEdge::zero;

    if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 2) {
      minus = makeGateDD<dd::GateMatrix>(dd::Xmat, n, lines.at(0));
      xp10 = makeGateDD<dd::GateMatrix>(dd::Xmat, n, control10, lines.at(0));
    }

    if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 3) {
      minus = makeGateDD<dd::TritMatrix>(dd::X3dag, n, lines.at(0));
      xp10 = makeGateDD<dd::TritMatrix>(dd::X3, n, control10, lines.at(0));
    }

    else if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 5) {
      minus = makeGateDD<dd::QuintMatrix>(dd::X5dag, n, lines.at(0));
      xp10 = makeGateDD<dd::QuintMatrix>(dd::X5, n, control10, lines.at(0));
    }

    state = multiply(cH, state);
    state = multiply(minus, state);
    state = multiply(xp10, state);

    return state;
  }
  vEdge spread3(QuantumRegisterCount n, std::vector<QuantumRegister> lines,
                vEdge& state) {
    dd::Controls const control01{{lines.at(0), 1}};
    auto cH = makeGateDD<dd::TritMatrix>(dd::H3(), n, control01, lines.at(1));

    dd::Controls const control10{{lines.at(1), 0}};
    mEdge minus = mEdge::zero;
    mEdge xp10 = mEdge::zero;

    if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 2) {
      minus = makeGateDD<dd::GateMatrix>(dd::Xmat, n, lines.at(0));
      xp10 = makeGateDD<dd::GateMatrix>(dd::Xmat, n, control10, lines.at(0));
    }

    if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 3) {
      minus = makeGateDD<dd::TritMatrix>(dd::X3dag, n, lines.at(0));
      xp10 = makeGateDD<dd::TritMatrix>(dd::X3, n, control10, lines.at(0));
    }

    else if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 5) {
      minus = makeGateDD<dd::QuintMatrix>(dd::X5dag, n, lines.at(0));
      xp10 = makeGateDD<dd::QuintMatrix>(dd::X5, n, control10, lines.at(0));
    }

    dd::Controls const control12{{lines.at(1), 2}};
    auto xp12 = makeGateDD<dd::TritMatrix>(dd::X3, n, control12, lines.at(2));
    auto csum21 = csum(n, lines.at(2), lines.at(1), true);

    state = multiply(cH, state);
    state = multiply(minus, state);
    state = multiply(xp10, state);
    state = multiply(xp12, state);
    state = multiply(csum21, state);
    state = multiply(csum21, state);

    return state;
  }
  vEdge spread5(QuantumRegisterCount n, std::vector<QuantumRegister> lines,
                vEdge& state) {
    dd::Controls const control01{{lines.at(0), 1}};
    auto cH = makeGateDD<dd::QuintMatrix>(dd::H5(), n, control01, lines.at(1));

    dd::Controls const control10{{lines.at(1), 0}};
    mEdge minus = mEdge::zero;
    mEdge xp10 = mEdge::zero;

    if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 2) {
      minus = makeGateDD<dd::GateMatrix>(dd::Xmat, n, lines.at(0));
      xp10 = makeGateDD<dd::GateMatrix>(dd::Xmat, n, control10, lines.at(0));
    }

    if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 3) {
      minus = makeGateDD<dd::TritMatrix>(dd::X3dag, n, lines.at(0));
      xp10 = makeGateDD<dd::TritMatrix>(dd::X3, n, control10, lines.at(0));
    }

    else if (registersSizes.at(static_cast<std::size_t>(lines.at(0))) == 5) {
      minus = makeGateDD<dd::QuintMatrix>(dd::X5dag, n, lines.at(0));
      xp10 = makeGateDD<dd::QuintMatrix>(dd::X5, n, control10, lines.at(0));
    }

    dd::Controls const control12{{lines.at(1), 2}};
    auto xp12 = makeGateDD<dd::QuintMatrix>(dd::X5, n, control12, lines.at(2));
    dd::Controls const control13{{lines.at(1), 3}};
    auto xp13 = makeGateDD<dd::QuintMatrix>(dd::X5, n, control13, lines.at(3));
    dd::Controls const control14{{lines.at(1), 4}};
    auto xp14 = makeGateDD<dd::QuintMatrix>(dd::X5, n, control14, lines.at(4));

    auto csum21 = csum(n, lines.at(2), lines.at(1), true);
    auto csum31 = csum(n, lines.at(3), lines.at(1), true);
    auto csum41 = csum(n, lines.at(4), lines.at(1), true);

    state = multiply(cH, state);
    state = multiply(minus, state);

    state = multiply(xp10, state);

    state = multiply(xp12, state);

    state = multiply(csum21, state);
    state = multiply(csum21, state);

    state = multiply(xp13, state);

    state = multiply(csum31, state);
    state = multiply(csum31, state);
    state = multiply(csum31, state);

    state = multiply(xp14, state);

    state = multiply(csum41, state);
    state = multiply(csum41, state);
    state = multiply(csum41, state);
    state = multiply(csum41, state);

    return state;
  }

public:
  template <class Edge>
  unsigned int nodeCount(const Edge& e,
                         std::unordered_set<decltype(e.nextNode)>& v) const {
    v.insert(e.nextNode);
    unsigned int sum = 1;
    if (!e.isTerminal()) {
      for (const auto& edge : e.nextNode->edges) {
        if (edge.nextNode != nullptr && !v.count(edge.nextNode)) {
          sum += nodeCount(edge, v);
        }
      }
    }
    return sum;
  }

  ///
  /// Vector and matrix extraction from DDs
  ///
public:
  /// Get a single element of the vector or matrix represented by the dd
  /// with root edge e \tparam Edge type of edge to use (vector or matrix)
  /// \param e edge to traverse
  /// \param path_elements string {0, 1, 2, 3}^n describing which outgoing
  /// edge should be followed
  ///        (for vectors entries are limited to 0 and 1)
  ///        If string is longer than required, the additional characters
  ///        are ignored.
  /// \return the complex amplitude of the specified element

  template <class Edge>
  ComplexValue getValueByPath(const Edge& edge,
                              const std::string& pathElements) {
    if (edge.isTerminal()) {
      return {CTEntry::val(edge.weight.real), CTEntry::val(edge.weight.img)};
    }

    auto tempCompNumb = complexNumber.getTemporary(1, 0);
    auto currentEdge = edge;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      ComplexNumbers::mul(tempCompNumb, tempCompNumb, currentEdge.weight);
      const auto tmp =
          static_cast<std::size_t>(pathElements.at(static_cast<std::size_t>(
                                       currentEdge.nextNode->varIndx)) -
                                   '0');
      assert(tmp <= currentEdge.nextNode->edges.size());
      currentEdge = currentEdge.nextNode->edges.at(tmp);
    } while (!currentEdge.isTerminal());

    ComplexNumbers::mul(tempCompNumb, tempCompNumb, currentEdge.weight);

    return {CTEntry::val(tempCompNumb.real), CTEntry::val(tempCompNumb.img)};
  }

  ComplexValue getValueByPath(const vEdge& edge,
                              std::vector<std::size_t>& reprI) {
    if (edge.isTerminal()) {
      return {CTEntry::val(edge.weight.real), CTEntry::val(edge.weight.img)};
    }
    return getValueByPath(edge, Complex::one, reprI);
  }

  ComplexValue getValueByPath(const vEdge& edge, const Complex& amp,
                              std::vector<std::size_t>& repr) {
    auto cNumb = complexNumber.mulCached(edge.weight, amp);

    if (edge.isTerminal()) {
      complexNumber.returnToCache(cNumb);
      return {CTEntry::val(cNumb.real), CTEntry::val(cNumb.img)};
    }

    ComplexValue returnAmp{};

    if (!edge.nextNode->edges.at(repr.front()).weight.approximatelyZero()) {
      std::vector<std::size_t> reprSlice(repr.begin() + 1, repr.end());
      returnAmp = getValueByPath(edge.nextNode->edges.at(repr.front()), cNumb,
                                 reprSlice);
    }

    complexNumber.returnToCache(cNumb);
    return returnAmp;
  }

  ComplexValue getValueByPath(const mEdge& edge,
                              std::vector<std::size_t>& reprI,
                              std::vector<std::size_t>& reprJ) {
    if (edge.isTerminal()) {
      return {CTEntry::val(edge.weight.real), CTEntry::val(edge.weight.img)};
    }
    return getValueByPath(edge, Complex::one, reprI, reprJ);
  }

  ComplexValue getValueByPath(const mEdge& edge, const Complex& amp,
                              std::vector<std::size_t>& reprI,
                              std::vector<std::size_t>& reprJ) {
    // row major encoding

    auto cNumb = complexNumber.mulCached(edge.weight, amp);

    if (edge.isTerminal()) {
      complexNumber.returnToCache(cNumb);
      return {CTEntry::val(cNumb.real), CTEntry::val(cNumb.img)};
    }

    const auto row = reprI.front();
    const auto col = reprJ.front();
    const auto rowMajorIndex = row * edge.nextNode->edges.size() + col;
    ComplexValue returnAmp{};

    if (!edge.nextNode->edges.at(rowMajorIndex).weight.approximatelyZero()) {
      std::vector<std::size_t> reprSliceI(reprI.begin() + 1, reprI.end());
      std::vector<std::size_t> reprSliceJ(reprJ.begin() + 1, reprJ.end());
      returnAmp = getValueByPath(edge.nextNode->edges.at(rowMajorIndex), cNumb,
                                 reprSliceI, reprSliceJ);
    }
    complexNumber.returnToCache(cNumb);
    return returnAmp;
  }

  CVec getVector(const vEdge& edge) {
    const auto dim = static_cast<const size_t>(std::accumulate(
        registersSizes.begin(), registersSizes.end(), 1, std::multiplies<>()));
    // allocate resulting vector
    auto vec = CVec(dim, {0.0, 0.0});
    getVector(edge, Complex::one, 0, vec, dim);
    return vec;
  }

  CVec getVectorizedMatrix(const mEdge& edge) {
    std::size_t dim = 1U;

    for (const auto registersSize : registersSizes) {
      dim = dim * registersSize * registersSize;
    }
    // allocate resulting vector
    auto vec = CVec(dim, {0.0, 0.0});
    getVector(edge, Complex::one, 0, vec, dim);
    return vec;
  }

  template <class Node>
  void getVector(const Edge<Node>& edge, const Complex& amp, std::size_t i,
                 CVec& vec, std::size_t next,
                 std::vector<std::size_t> pathTracker = {}) {
    // calculate new accumulated amplitude
    auto cNumb = complexNumber.mulCached(edge.weight, amp);

    // base case
    if (edge.isTerminal()) {
      if (std::is_same<Node, mNode>::value) {
        for (const auto j : pathTracker) {
          std::cout << j;
        }
        std::cout << ": ";
        std::cout << cNumb << std::endl;
      }
      vec.at(i) = {CTEntry::val(cNumb.real), CTEntry::val(cNumb.img)};
      complexNumber.returnToCache(cNumb);
      return;
    }

    auto offset = (next - i) / edge.nextNode->edges.size();

    for (auto k = 0UL; k < edge.nextNode->edges.size(); k++) {
      if (std::is_same<Node, mNode>::value) {
        pathTracker.push_back(k);
        getVector(edge.nextNode->edges.at(k), cNumb, i + (k * offset), vec,
                  i + ((k + 1) * offset), pathTracker);
        pathTracker.pop_back();
      } else {
        if (!edge.nextNode->edges.at(k).weight.approximatelyZero()) {
          getVector(edge.nextNode->edges.at(k), cNumb, i + (k * offset), vec,
                    i + ((k + 1) * offset));
        }
      }
    }

    complexNumber.returnToCache(cNumb);
  }

  std::vector<std::size_t> getReprOfIndex(const std::size_t i,
                                          const std::size_t numEntries) {
    std::vector<std::size_t> repr;
    repr.reserve(numberOfQuantumRegisters);
    // get representation
    auto iIndex = i;
    auto pathWay = 0UL;
    auto cardinality = numEntries;

    auto counter = 0UL;
    auto index = 0UL;

    while (counter < numberOfQuantumRegisters) {
      try {
        index = numberOfQuantumRegisters - counter - 1;
        cardinality = cardinality / registersSizes.at(index);
        pathWay = iIndex / cardinality;
        iIndex = iIndex % cardinality;
      } catch (const std::exception& e) {
        std::cout << "index = " << index << ", cardinality = " << cardinality
                  << " ,counter = " << counter << std::endl;
      }

      repr.push_back(pathWay);
      counter = counter + 1;
    }

    return repr;
  }

  void printVector(const vEdge& edge, bool nonZero = false) {
    // unsigned long long numEntries = static_cast<unsigned long long
    // int>(std::accumulate(registersSizes.begin(), registersSizes.end(),
    // 1,std::multiplies<>()));

    std::size_t numEntries = 1ULL;
    for (const auto registersSize : registersSizes) {
      numEntries = numEntries * registersSize;
    }

    for (auto i = 0ULL; i < numEntries; i++) {
      auto reprI = getReprOfIndex(i, numEntries);
      // get amplitude
      const auto amplitude = getValueByPath(edge, reprI);

      if (!amplitude.approximatelyZero() || !nonZero) {
        for (const auto coeff : reprI) {
          std::cout << coeff;
        }
        reprI.clear();

        constexpr auto precision = 3;
        // set fixed width to maximum of a printed number
        // (-) 0.precision plus/minus 0.precision i
        constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
        std::cout << ": " << std::setw(width)
                  << ComplexValue::toString(amplitude.r, amplitude.i, false,
                                            precision)
                  << "\n";
      }
    }
    std::cout << std::flush;
  }

  static void printComplexVector(const CVec& vector) {
    for (auto i = 0ULL; i < vector.size(); i++) {
      std::cout << i;

      constexpr auto precision = 3;
      // set fixed width to maximum of a printed number
      // (-) 0.precision plus/minus 0.precision i
      constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
      std::cout << ": " << std::setw(width) << vector.at(i) << "\n";
    }
    std::cout << std::flush;
  }

private:
  // check whether node represents a symmetric matrix or the identity
  void checkSpecialMatrices(mNode* node) {
    if (node->varIndx == -1) {
      return;
    }

    node->identity = false;  // assume not identity
    node->symmetric = false; // assume symmetric

    // check if matrix is symmetric
    auto basicDim = registersSizes.at(static_cast<std::size_t>(node->varIndx));

    for (auto i = 0UL; i < basicDim; i++) {
      if (!node->edges.at(i * basicDim + i).nextNode->symmetric) {
        return;
      }
    }
    // TODO WHY RETURN IF DIAGONAL IS SYMMETRIC??
    // if (!node->edges.at(0).nextNode->symmetric ||
    // !node->edges.at(3).nextNode->symmetric) return;

    for (auto i = 0UL; i < basicDim; i++) {
      for (auto j = 0UL; j < basicDim; j++) {
        if (i != j) {
          // row major indexing - enable optimization here
          if (transpose(node->edges.at(i * basicDim + j)) !=
              node->edges.at(j * basicDim + i)) {
            return;
          }
        }
      }
    }
    // if (transpose(node->edges.at(1)) != node->edges.at(2)) return;

    node->symmetric = true;

    // check if matrix resembles identity
    for (auto i = 0UL; i < basicDim; i++) {
      for (auto j = 0UL; j < basicDim; j++) {
        // row major indexing - enable optimization here
        if (i == j) {
          if (!(node->edges[i * basicDim + j].nextNode->identity) ||
              (node->edges[i * basicDim + j].weight) != Complex::one) {
            return;
          }
        } else {
          if ((node->edges[i * basicDim + j].weight) != Complex::zero) {
            return;
          }
        }
      }
    }
    /*
  if (!(p->e[0].p->ident) || (p->e[1].w) != Complex::zero ||
      (p->e[2].w) != Complex::zero || (p->e[0].w) != Complex::one ||
      (p->e[3].w) != Complex::one || !(p->e[3].p->ident))
      return;
  */
    node->identity = true;
  }

  ///
  /// Matrix (conjugate) transpose
  ///
public:
  // todo figure out the parameters here
  UnaryComputeTable<mEdge, mEdge, 4096> matrixTranspose{};
  UnaryComputeTable<mEdge, mEdge, 4096> conjugateMatrixTranspose{};

  mEdge transpose(const mEdge& edge) {
    if (edge.nextNode == nullptr || edge.isTerminal() ||
        edge.nextNode->symmetric) {
      return edge;
    }

    // check in compute table
    auto result = matrixTranspose.lookup(edge);
    if (result.nextNode != nullptr) {
      return result;
    }

    std::vector<mEdge> newEdge{};
    auto basicDim =
        registersSizes.at(static_cast<std::size_t>(edge.nextNode->varIndx));

    // transpose sub-matrices and rearrange as required
    for (auto i = 0U; i < basicDim; i++) {
      for (auto j = 0U; j < basicDim; j++) {
        newEdge.at(basicDim * i + j) =
            transpose(edge.nextNode->edges.at(basicDim * j + i));
      }
    }
    // create new top node
    result = makeDDNode(edge.nextNode->varIndx, newEdge);
    // adjust top weight
    auto c = complexNumber.getTemporary();
    ComplexNumbers::mul(c, result.weight, edge.weight);
    result.weight = complexNumber.lookup(c);

    // put in compute table
    matrixTranspose.insert(edge, result);
    return result;
  }
  mEdge conjugateTranspose(const mEdge& edge) {
    if (edge.nextNode == nullptr) {
      return edge;
    }
    if (edge.isTerminal()) { // terminal case
      auto result = edge;
      result.weight = ComplexNumbers::conj(edge.weight);
      return result;
    }

    // check if in compute table
    auto result = conjugateMatrixTranspose.lookup(edge);
    if (result.nextNode != nullptr) {
      return result;
    }

    std::vector<mEdge> newEdge(edge.nextNode->edges.size(),
                               dd::Edge<mNode>::zero);
    auto basicDim =
        registersSizes.at(static_cast<std::size_t>(edge.nextNode->varIndx));

    // conjugate transpose submatrices and rearrange as required
    for (auto i = 0U; i < basicDim; ++i) {
      for (auto j = 0U; j < basicDim; ++j) {
        newEdge.at(basicDim * i + j) =
            conjugateTranspose(edge.nextNode->edges.at(basicDim * j + i));
      }
    }
    // create new top node
    result = makeDDNode(edge.nextNode->varIndx, newEdge);

    auto c = complexNumber.getTemporary();
    // adjust top weight including conjugate
    ComplexNumbers::mul(c, result.weight, ComplexNumbers::conj(edge.weight));
    result.weight = complexNumber.lookup(c);

    // put it in the compute table
    conjugateMatrixTranspose.insert(edge, result);
    return result;
  }

  ///
  /// Unique tables, Reference counting and garbage collection
  ///
public:
  // unique tables
  template <class Node> [[nodiscard]] UniqueTable<Node>& getUniqueTable();

  template <class Node> void incRef(const Edge<Node>& e) {
    getUniqueTable<Node>().incRef(e);
  }

  template <class Node> void decRef(const Edge<Node>& e) {
    getUniqueTable<Node>().decRef(e);
  }

  UniqueTable<vNode> vUniqueTable{numberOfQuantumRegisters};
  UniqueTable<mNode> mUniqueTable{numberOfQuantumRegisters};
};

inline void clearUniqueTables() {
  // TODO IMPLEMENT
  // vUniqueTable.clear();
  // mUniqueTable.clear();
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline MDDPackage::vNode MDDPackage::vNode::terminalNode{
    {{{nullptr, Complex::zero}, {nullptr, Complex::zero}}}, nullptr, 0, -1};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline MDDPackage::mNode MDDPackage::mNode::terminalNode{
    {{{nullptr, Complex::zero},
      {nullptr, Complex::zero},
      {nullptr, Complex::zero},
      {nullptr, Complex::zero}}},
    nullptr,
    0,
    -1,
    true,
    true};

template <>
[[nodiscard]] inline UniqueTable<MDDPackage::vNode>&
MDDPackage::getUniqueTable() {
  return vUniqueTable;
}

template <>
[[nodiscard]] inline UniqueTable<MDDPackage::mNode>&
MDDPackage::getUniqueTable() {
  return mUniqueTable;
}

template <>
[[nodiscard]] inline ComputeTable<
    MDDPackage::vCachedEdge, MDDPackage::vCachedEdge, MDDPackage::vCachedEdge>&
MDDPackage::getAddComputeTable() {
  return vectorAdd;
}

template <>
[[nodiscard]] inline ComputeTable<
    MDDPackage::mCachedEdge, MDDPackage::mCachedEdge, MDDPackage::mCachedEdge>&
MDDPackage::getAddComputeTable() {
  return matrixAdd;
}

template <>
[[nodiscard]] inline ComputeTable<MDDPackage::mEdge, MDDPackage::vEdge,
                                  MDDPackage::vCachedEdge>&
MDDPackage::getMultiplicationComputeTable() {
  return matrixVectorMultiplication;
}

template <>
[[nodiscard]] inline ComputeTable<MDDPackage::mEdge, MDDPackage::mEdge,
                                  MDDPackage::mCachedEdge>&
MDDPackage::getMultiplicationComputeTable() {
  return matrixMatrixMultiplication;
}

template <>
[[nodiscard]] inline ComputeTable<MDDPackage::vEdge, MDDPackage::vEdge,
                                  MDDPackage::vCachedEdge, 4096>&
MDDPackage::getKroneckerComputeTable() {
  return vectorKronecker;
}

template <>
[[nodiscard]] inline ComputeTable<MDDPackage::mEdge, MDDPackage::mEdge,
                                  MDDPackage::mCachedEdge, 4096>&
MDDPackage::getKroneckerComputeTable() {
  return matrixKronecker;
}

} // namespace dd

#endif
