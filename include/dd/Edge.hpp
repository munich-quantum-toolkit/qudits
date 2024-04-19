/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_EDGE_HPP
#define DD_PACKAGE_EDGE_HPP

#include "Complex.hpp"
#include "ComplexValue.hpp"

#include <cstddef>
#include <utility>

namespace dd {
template <class Node> struct Edge {
  Node* nextNode;
  Complex weight;

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close enough"
  /// according to a given tolerance this notion of equivalence is chosen to
  /// counter floating point inaccuracies
  constexpr bool operator==(const Edge& other) const {
    return nextNode == other.nextNode &&
           weight.approximatelyEquals(other.weight);
  }
  constexpr bool operator!=(const Edge& other) const {
    return !operator==(other);
  }

  [[nodiscard]] constexpr bool isTerminal() const {
    return Node::isTerminal(nextNode);
  }

  // edges pointing to zero and one terminals
  static const inline Edge one{
      Node::terminal,
      Complex::one}; // NOLINT(readability-identifier-naming) automatic renaming
                     // does not work reliably, so skip linting
  static const inline Edge zero{
      Node::terminal,
      Complex::zero}; // NOLINT(readability-identifier-naming) automatic
                      // renaming does not work reliably, so skip linting

  [[nodiscard]] static constexpr Edge terminal(const Complex& weight) {
    return {Node::terminal, weight};
  }
  [[nodiscard]] constexpr bool isZeroTerminal() const {
    return Node::isTerminal(nextNode) && weight == Complex::zero;
  }
  [[nodiscard]] constexpr bool isOneTerminal() const {
    return Node::isTerminal(nextNode) && weight == Complex::one;
  }
};

template <typename Node> struct CachedEdge {
  Node* nextNode{};
  ComplexValue weight{};

  CachedEdge() = default;
  CachedEdge(Node* nextNode, const ComplexValue& weightOriginal)
      : nextNode(nextNode), weight(weightOriginal) {}
  CachedEdge(Node* nextNode, const Complex& weightComplexNumber)
      : nextNode(nextNode) {
    weight.r = CTEntry::val(weightComplexNumber.real);
    weight.i = CTEntry::val(weightComplexNumber.img);
  }

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close
  /// enough" according to a given tolerance this notion of equivalence is
  /// chosen to counter floating point inaccuracies
  bool operator==(const CachedEdge& other) const {
    return nextNode == other.nextNode &&
           weight.approximatelyEquals(other.weight);
  }
  bool operator!=(const CachedEdge& other) const { return !operator==(other); }
};
} // namespace dd

namespace std {
template <class Node> struct hash<dd::Edge<Node>> {
  std::size_t operator()(dd::Edge<Node> const& edge) const noexcept {
    auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(edge.nextNode));
    auto h2 = std::hash<dd::Complex>{}(edge.weight);
    return dd::combineHash(h1, h2);
  }
};

template <class Node> struct hash<dd::CachedEdge<Node>> {
  std::size_t operator()(dd::CachedEdge<Node> const& edge) const noexcept {
    auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(edge.nextNode));
    auto h2 = std::hash<dd::ComplexValue>{}(edge.weight);
    return dd::combineHash(h1, h2);
  }
};
} // namespace std

#endif // DD_PACKAGE_EDGE_HPP
