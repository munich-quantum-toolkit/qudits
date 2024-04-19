/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_DATATYPES_HPP
#define DDpackage_DATATYPES_HPP

#include "MDDPackage.hpp"

#include <complex>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dd {
// integer type used for indexing QuantumRegisters
// needs to be a signed type to encode -1 as the index for the terminal
// std::int_fast8_t can at least address 128 QuantumRegisters as [0, ..., 127]
// TODO understand how many quantum registers to put in a circuit, depending on
// the dimensions

using QuantumRegister = std::int_fast8_t;

static_assert(std::is_signed_v<QuantumRegister>,
              "Type QuantumRegister must be signed.");

// integer type used for specifying numbers of QuantumRegisters
using QuantumRegisterCount = std::make_unsigned<QuantumRegister>::type;

// integer type used for reference counting
// 32bit suffice for a max ref count of around 4 billion
using RefCount = std::uint_fast32_t;
static_assert(std::is_unsigned_v<RefCount>, "RefCount should be unsigned.");

// floating point type to use
using fp = double;
static_assert(
    std::is_floating_point_v<fp>,
    "fp should be a floating point type (float, *double*, long double)");

//-----------------------------------------------------------------------------------------
// TODO BELOW TEMPORARY LEGACY CODE
// Gate matrices
static constexpr std::uint_fast8_t RADIX_2 = 2;
// max no. of edges = RADIX_3^2
static constexpr std::uint_fast8_t EDGE2 = RADIX_2 * RADIX_2;
// Gate matrices
static constexpr std::uint_fast8_t RADIX_3 = 3;
// max no. of edges = RADIX_3^2
static constexpr std::uint_fast8_t EDGE3 = RADIX_3 * RADIX_3;

static constexpr std::uint_fast8_t RADIX_4 = 4;

static constexpr std::uint_fast8_t EDGE4 = RADIX_4 * RADIX_4;

static constexpr std::uint_fast8_t RADIX_5 = 5;

static constexpr std::uint_fast8_t EDGE5 = RADIX_5 * RADIX_5;

static constexpr std::uint_fast8_t RADIX_6 = 6;

static constexpr std::uint_fast8_t EDGE6 = RADIX_6 * RADIX_6;

static constexpr std::uint_fast8_t RADIX_7 = 7;

static constexpr std::uint_fast8_t EDGE7 = RADIX_7 * RADIX_7;
// TODO ABOVE TEMPORARY LEGACY CODE
//-----------------------------------------------------------------------------------------

enum class BasisStates { Zero, One, Plus, Minus, Right, Left };

static constexpr fp SQRT2_2 = static_cast<fp>(
    0.707106781186547524400844362104849039284835937688474036588L);
static constexpr fp SQRT3_3 = static_cast<fp>(
    0.577350269189625764509148780501957455647601751270126876018L);
static constexpr fp SQRT4_4 = static_cast<fp>(
    0.500000000000000000000000000000000000000000000000000000000L);
static constexpr fp SQRT5_5 = static_cast<fp>(
    0.447213595499957939281834733746255247088123671922305144854L);
static constexpr fp SQRT6_6 = static_cast<fp>(
    0.408248290463863016366214012450981898660991246776111688072L);
static constexpr fp SQRT7_7 = static_cast<fp>(
    0.377964473009227227214516536234180060815751311868921454338L);

static constexpr fp PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);
static constexpr fp PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
static constexpr fp PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);

using CVec = std::vector<std::complex<fp>>;
using CMat = std::vector<CVec>;

// use hash maps for representing sparse vectors of probabilities
using ProbabilityVector = std::unordered_map<std::size_t, fp>;

static constexpr std::uint_least64_t SERIALIZATION_VERSION = 1;

// 64bit mixing hash (from MurmurHash3,
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)
constexpr std::size_t murmur64(std::size_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

// combine two 64bit hashes into one 64bit hash (boost::hash_combine,
// https://www.boost.org/LICENSE_1_0.txt)
constexpr std::size_t combineHash(std::size_t lhs, std::size_t rhs) {
  lhs ^= rhs + 0x9e3779b97f4a7c15ULL + (lhs << 6) + (lhs >> 2);
  return lhs;
}

// alternative hash combinator (from Google's city hash,
// https://github.com/google/cityhash/blob/master/COPYING)
//    constexpr std::size_t combineHash(std::size_t lhs, std::size_t rhs) {
//        const std::size_t kMul = 0x9ddfea08eb382d69ULL;
//        std::size_t a = (lhs ^ rhs) * kMul;
//        a ^= (a >> 47);
//        std::size_t b = (rhs ^ a) * kMul;
//        b ^= (b >> 47);
//        b *= kMul;
//        return b;
//    }

// calculates the Units in Last Place (ULP) distance of two floating point
// numbers
[[maybe_unused]] static std::size_t ulpDistance(fp a, fp b) {
  if (a == b) {
    return 0;
  }

  std::size_t ulps = 1;
  fp nextFP = std::nextafter(a, b);
  while (nextFP != b) {
    ulps++;
    nextFP = std::nextafter(nextFP, b);
  }
  return ulps;
}

} // namespace dd
#endif // DDpackage_DATATYPES_HPP
