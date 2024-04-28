/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_GATEMATRIXDEFINITIONS_H
#define DD_PACKAGE_GATEMATRIXDEFINITIONS_H

#include "ComplexValue.hpp"
#include "Definitions.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace dd {
// Complex constants
constexpr ComplexValue COMPLEX_ONE = {1., 0.};
constexpr ComplexValue COMPLEX_MONE = {-1., 0.};
constexpr ComplexValue COMPLEX_ZERO = {0., 0.};
constexpr ComplexValue COMPLEX_I = {0., 1.};
constexpr ComplexValue COMPLEX_MI = {0., -1.};
constexpr ComplexValue COMPLEX_SQRT2_2 = {SQRT2_2, 0.};
constexpr ComplexValue COMPLEX_MSQRT2_2 = {-SQRT2_2, 0.};
constexpr ComplexValue COMPLEX_ISQRT2_2 = {0., SQRT2_2};
constexpr ComplexValue COMPLEX_MISQRT2_2 = {0., -SQRT2_2};
constexpr ComplexValue COMPLEX_1PLUSI = {SQRT2_2, SQRT2_2};
constexpr ComplexValue COMPLEX_1MINUSI = {SQRT2_2, -SQRT2_2};
constexpr ComplexValue COMPLEX_1PLUSI_2 = {0.5, 0.5};
constexpr ComplexValue COMPLEX_1MINUSI_2 = {0.5, -0.5};

constexpr ComplexValue COMPLEX_SQRT3_3 = {SQRT3_3, 0.};
constexpr ComplexValue COMPLEX_SQRT4_4 = {SQRT4_4, 0.};
constexpr ComplexValue COMPLEX_SQRT5_5 = {SQRT5_5, 0.};
constexpr ComplexValue COMPLEX_SQRT6_6 = {SQRT6_6, 0.};
constexpr ComplexValue COMPLEX_SQRT7_7 = {SQRT7_7, 0.};

using GateMatrix = std::array<ComplexValue, EDGE2>;
using TritMatrix = std::array<ComplexValue, EDGE3>;
using QuartMatrix = std::array<ComplexValue, EDGE4>;
using QuintMatrix = std::array<ComplexValue, EDGE5>;
using SextMatrix = std::array<ComplexValue, EDGE6>;
using SeptMatrix = std::array<ComplexValue, EDGE7>;

// NOLINTBEGIN(readability-identifier-naming) As these constants are used by
// other projects, we keep the naming
constexpr GateMatrix Imat{COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
constexpr GateMatrix Hmat{COMPLEX_SQRT2_2, COMPLEX_SQRT2_2, COMPLEX_SQRT2_2,
                          COMPLEX_MSQRT2_2};
constexpr GateMatrix Xmat{COMPLEX_ZERO, COMPLEX_ONE, COMPLEX_ONE, COMPLEX_ZERO};
constexpr GateMatrix Ymat{COMPLEX_ZERO, COMPLEX_MI, COMPLEX_I, COMPLEX_ZERO};
constexpr GateMatrix Zmat{COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO,
                          COMPLEX_MONE};
constexpr GateMatrix Smat{COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_I};
constexpr GateMatrix Sdagmat{COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO,
                             COMPLEX_MI};
constexpr GateMatrix Tmat{COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO,
                          COMPLEX_1PLUSI};
constexpr GateMatrix Tdagmat{COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO,
                             COMPLEX_1MINUSI};
constexpr GateMatrix SXmat{COMPLEX_1PLUSI_2, COMPLEX_1MINUSI_2,
                           COMPLEX_1MINUSI_2, COMPLEX_1PLUSI_2};
constexpr GateMatrix SXdagmat{COMPLEX_1MINUSI_2, COMPLEX_1PLUSI_2,
                              COMPLEX_1PLUSI_2, COMPLEX_1MINUSI_2};
constexpr GateMatrix Vmat{COMPLEX_SQRT2_2, COMPLEX_MISQRT2_2, COMPLEX_MISQRT2_2,
                          COMPLEX_SQRT2_2};
constexpr GateMatrix Vdagmat{COMPLEX_SQRT2_2, COMPLEX_ISQRT2_2,
                             COMPLEX_ISQRT2_2, COMPLEX_SQRT2_2};

inline GateMatrix Pimat(size_t i) {
  GateMatrix zero = {COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};
  zero.at(i + i * 2) = COMPLEX_ONE;
  return zero;
}

inline GateMatrix U3mat(fp lambda, fp phi, fp theta) {
  return GateMatrix{{{std::cos(theta / 2.), 0.},
                     {-std::cos(lambda) * std::sin(theta / 2.),
                      -std::sin(lambda) * std::sin(theta / 2.)},
                     {std::cos(phi) * std::sin(theta / 2.),
                      std::sin(phi) * std::sin(theta / 2.)},
                     {std::cos(lambda + phi) * std::cos(theta / 2.),
                      std::sin(lambda + phi) * std::cos(theta / 2.)}}};
}

inline GateMatrix U2mat(fp lambda, fp phi) {
  return GateMatrix{
      COMPLEX_SQRT2_2,
      {-std::cos(lambda) * SQRT2_2, -std::sin(lambda) * SQRT2_2},
      {std::cos(phi) * SQRT2_2, std::sin(phi) * SQRT2_2},
      {std::cos(lambda + phi) * SQRT2_2, std::sin(lambda + phi) * SQRT2_2}};
}

inline GateMatrix Phasemat(fp lambda) {
  return GateMatrix{COMPLEX_ONE,
                    COMPLEX_ZERO,
                    COMPLEX_ZERO,
                    {std::cos(lambda), std::sin(lambda)}};
}

inline GateMatrix RXmat(fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {0., -std::sin(lambda / 2.)},
                     {0., -std::sin(lambda / 2.)},
                     {std::cos(lambda / 2.), 0.}}};
}

inline GateMatrix RYmat(fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {-std::sin(lambda / 2.), 0.},
                     {std::sin(lambda / 2.), 0.},
                     {std::cos(lambda / 2.), 0.}}};
}

inline GateMatrix RZmat(fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), -std::sin(lambda / 2.)},
                     COMPLEX_ZERO,
                     COMPLEX_ZERO,
                     {std::cos(lambda / 2.), std::sin(lambda / 2.)}}};
}

inline GateMatrix H() {
  return GateMatrix{COMPLEX_SQRT2_2, COMPLEX_SQRT2_2, COMPLEX_SQRT2_2,
                    COMPLEX_MSQRT2_2};
}
inline GateMatrix RXY(fp theta, fp phi) {
  GateMatrix rotation = {
      dd::ComplexValue{std::cos(theta / 2.), 0.},
      dd::ComplexValue{-std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)},
      dd::ComplexValue{std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)},
      dd::ComplexValue{std::cos(theta / 2.), 0.}};
  return rotation;
}

inline GateMatrix RH() {
  GateMatrix rotation = {COMPLEX_ISQRT2_2, COMPLEX_ISQRT2_2, COMPLEX_ISQRT2_2,
                         COMPLEX_MISQRT2_2};
  return rotation;
}

inline GateMatrix RZ(fp phi) {
  GateMatrix rotation = {
      dd::ComplexValue{std::cos(phi / 2), -std::sin(phi / 2)}, COMPLEX_ZERO,
      COMPLEX_ZERO, dd::ComplexValue{std::cos(phi / 2), std::sin(phi / 2)}};
  return rotation;
}

inline GateMatrix VirtRZ(fp phi, size_t i) {
  GateMatrix zero = {COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  zero.at(i + i * 2) = dd::ComplexValue{std::cos(phi), -std::sin(phi)};
  return zero;
}

inline GateMatrix embX2(fp phi) {
  GateMatrix identity = {COMPLEX_ONE, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(0) = COMPLEX_ZERO;
  identity.at(1) = dd::ComplexValue{-std::sin(phi), -std::cos(phi)};
  identity.at(2) = dd::ComplexValue{std::sin(phi), -std::cos(phi)};
  identity.at(3) = COMPLEX_ZERO;
  return identity;
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

constexpr TritMatrix I3{COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

inline TritMatrix Pi3(size_t i) {
  TritMatrix zero = {COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                     COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                     COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};
  zero.at(i + i * 3) = COMPLEX_ONE;
  return zero;
}

inline TritMatrix H3() {
  return TritMatrix{COMPLEX_SQRT3_3,
                    COMPLEX_SQRT3_3,
                    COMPLEX_SQRT3_3,

                    COMPLEX_SQRT3_3,
                    COMPLEX_SQRT3_3 * dd::ComplexValue{std::cos(2. * PI / 3.),
                                                       std::sin(2. * PI / 3.)},
                    COMPLEX_SQRT3_3 * dd::ComplexValue{std::cos(4. * PI / 3.),
                                                       std::sin(4. * PI / 3.)},

                    COMPLEX_SQRT3_3,
                    COMPLEX_SQRT3_3 * dd::ComplexValue{std::cos(4. * PI / 3.),
                                                       std::sin(4. * PI / 3.)},
                    COMPLEX_SQRT3_3 * dd::ComplexValue{std::cos(2. * PI / 3.),
                                                       std::sin(2. * PI / 3.)}};
}

constexpr TritMatrix X3dag{COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
                           COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO};

constexpr TritMatrix X3{COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
                        COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO};

constexpr TritMatrix PI02{COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_MONE,
                          COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                          COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO};
constexpr TritMatrix P_I02DAG{COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
                              COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                              COMPLEX_MONE, COMPLEX_ZERO, COMPLEX_ZERO};
constexpr TritMatrix X01{COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                         COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
constexpr TritMatrix Z01{COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_MONE, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

inline TritMatrix U301(fp lambda, fp phi, fp theta) {
  return TritMatrix{{{std::cos(theta / 2.), 0.},
                     {-std::cos(lambda) * std::sin(theta / 2.),
                      -std::sin(lambda) * std::sin(theta / 2.)},
                     COMPLEX_ZERO,
                     {std::cos(phi) * std::sin(theta / 2.),
                      std::sin(phi) * std::sin(theta / 2.)},
                     {std::cos(lambda + phi) * std::cos(theta / 2.),
                      std::sin(lambda + phi) * std::cos(theta / 2.)},
                     COMPLEX_ZERO,
                     COMPLEX_ZERO,
                     COMPLEX_ZERO,
                     COMPLEX_ONE}};
}

inline TritMatrix RXY3(fp theta, fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 2 or levb > 2) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  TritMatrix identity = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(3 * leva + leva) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  identity.at(3 * leva + levb) =
      dd::ComplexValue{-std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(3 * levb + leva) =
      dd::ComplexValue{std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(3 * levb + levb) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  return identity;
}

inline TritMatrix RH3(size_t leva, size_t levb) {
  if (leva > levb or leva > 2 or levb > 2) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  TritMatrix identity = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(3 * leva + leva) = COMPLEX_ISQRT2_2;
  identity.at(3 * leva + levb) = COMPLEX_ISQRT2_2;
  identity.at(3 * levb + leva) = COMPLEX_ISQRT2_2;
  identity.at(3 * levb + levb) = COMPLEX_MISQRT2_2;
  return identity;
}

inline TritMatrix RZ3(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 2 or levb > 2) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  TritMatrix identity = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(3 * leva + leva) =
      dd::ComplexValue{std::cos(phi / 2), -std::sin(phi / 2)};
  identity.at(3 * levb + levb) =
      dd::ComplexValue{std::cos(phi / 2), +std::sin(phi / 2)};
  return identity;
}

inline TritMatrix VirtRZ3(fp phi, size_t i) {
  TritMatrix zero = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                     COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                     COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  zero.at(i + i * 3) = dd::ComplexValue{std::cos(phi), -std::sin(phi)};
  return zero;
}

inline TritMatrix Z3() {
  TritMatrix id = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                   COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                   COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  for (int level = 0; level < 3; ++level) {
    const double angle = fmod(2.0 * level / 3, 2.0) * PI;
    id.at(level + level * 3) = dd::ComplexValue{cos(angle), sin(angle)};
  }

  return id;
}
inline TritMatrix S3() {
  TritMatrix id = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                   COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                   COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  for (int level = 0; level < 3; ++level) {
    const double omegaArg = fmod(2.0 / 3 * level * (level + 1) / 2.0, 2.0);
    const auto omega = dd::ComplexValue{cos(omegaArg * PI), sin(omegaArg * PI)};
    id.at(level + level * 3) = omega;
  }
  return id;
}

inline TritMatrix embX3(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 2 or levb > 2) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  TritMatrix identity = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(3 * leva + leva) = COMPLEX_ZERO;
  identity.at(3 * leva + levb) =
      dd::ComplexValue{-std::sin(phi), -std::cos(phi)};
  identity.at(3 * levb + leva) =
      dd::ComplexValue{std::sin(phi), -std::cos(phi)};
  identity.at(3 * levb + levb) = COMPLEX_ZERO;
  return identity;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

constexpr QuartMatrix I4{COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                         COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

inline QuartMatrix Pi4(size_t i) {
  QuartMatrix zero = {COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};
  zero.at(i + i * 4) = COMPLEX_ONE;
  return zero;
}

inline QuartMatrix H4() {
  return QuartMatrix{
      COMPLEX_SQRT4_4,
      COMPLEX_SQRT4_4,
      COMPLEX_SQRT4_4,
      COMPLEX_SQRT4_4,

      COMPLEX_SQRT4_4,
      COMPLEX_SQRT4_4 *
          dd::ComplexValue{std::cos(2. * PI / 4.), std::sin(2. * PI / 4.)},
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(2. * 2. * PI / 4.),
                                         std::sin(2. * 2. * PI / 4.)},
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(3. * 2. * PI / 4.),
                                         std::sin(3. * 2. * PI / 4.)},

      COMPLEX_SQRT4_4,
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(2. * 2. * PI / 4.),
                                         std::sin(2. * 2. * PI / 4.)},
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(4. * 2. * PI / 4.),
                                         std::sin(4. * 2. * PI / 4.)},
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(6. * 2. * PI / 4.),
                                         std::sin(6. * 2. * PI / 4.)},

      COMPLEX_SQRT4_4,
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(3. * 2. * PI / 4.),
                                         std::sin(3. * 2. * PI / 4.)},
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(6. * 2. * PI / 4.),
                                         std::sin(6. * 2. * PI / 4.)},
      COMPLEX_SQRT4_4 * dd::ComplexValue{std::cos(9. * 2. * PI / 4.),
                                         std::sin(9. * 2. * PI / 4.)}};
}
inline QuartMatrix RXY4(fp theta, fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 3 or levb > 3) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuartMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(4 * leva + leva) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  identity.at(4 * leva + levb) =
      dd::ComplexValue{-std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(4 * levb + leva) =
      dd::ComplexValue{std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(4 * levb + levb) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  return identity;
}

inline QuartMatrix RH4(size_t leva, size_t levb) {
  if (leva > levb or leva > 3 or levb > 3) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuartMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(4 * leva + leva) = COMPLEX_ISQRT2_2;
  identity.at(4 * leva + levb) = COMPLEX_ISQRT2_2;
  identity.at(4 * levb + leva) = COMPLEX_ISQRT2_2;
  identity.at(4 * levb + levb) = COMPLEX_MISQRT2_2;
  return identity;
}
constexpr QuartMatrix X4dag{
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};

constexpr QuartMatrix X4{
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO};

inline QuartMatrix RZ4(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 3 or levb > 3) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuartMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(4 * leva + leva) =
      dd::ComplexValue{std::cos(phi / 2), -std::sin(phi / 2)};
  identity.at(4 * levb + levb) =
      dd::ComplexValue{std::cos(phi / 2), +std::sin(phi / 2)};
  return identity;
}

inline QuartMatrix VirtRZ4(fp phi, size_t i) {
  QuartMatrix zero = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  zero.at(i + i * 4) = dd::ComplexValue{std::cos(phi), -std::sin(phi)};
  return zero;
}

inline QuartMatrix Z4() {
  QuartMatrix id = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

  for (auto level = 0; level < 4; level++) {
    const double angle = fmod(2.0 * level / 4, 2.0) * PI;
    id.at(level + level * 4) =
        dd::ComplexValue{std::cos(angle), std::sin(angle)};
  }

  return id;
}
inline QuartMatrix S4() {
  QuartMatrix id = {COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  for (auto level = 0; level < 4; level++) {
    const double omegaArg = fmod(2.0 / 4 * level * (level + 1) / 2.0, 2.0);
    const auto omega =
        dd::ComplexValue{std::cos(omegaArg * PI), std::sin(omegaArg * PI)};
    id.at(level + level * 4) = omega;
  }
  return id;
}

inline QuartMatrix embX4(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 3 or levb > 3) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuartMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(4 * leva + leva) = COMPLEX_ZERO;
  identity.at(4 * leva + levb) =
      dd::ComplexValue{-std::sin(phi), -std::cos(phi)};
  identity.at(4 * levb + leva) =
      dd::ComplexValue{std::sin(phi), -std::cos(phi)};
  identity.at(4 * levb + levb) = COMPLEX_ZERO;
  return identity;
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

constexpr QuintMatrix I5{
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

inline QuintMatrix Pi5(size_t i) {
  QuintMatrix zero = {
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};
  zero.at(i + i * 5) = COMPLEX_ONE;
  return zero;
}
inline QuintMatrix H5() {
  return QuintMatrix{
      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5,

      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5 *
          dd::ComplexValue{std::cos(2. * PI / 5.), std::sin(2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(2. * 2. * PI / 5.),
                                         std::sin(2. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(3. * 2. * PI / 5.),
                                         std::sin(3. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(4. * 2. * PI / 5.),
                                         std::sin(4. * 2. * PI / 5.)},

      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(2. * 2. * PI / 5.),
                                         std::sin(2. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(4. * 2. * PI / 5.),
                                         std::sin(4. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(6. * 2. * PI / 5.),
                                         std::sin(6. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(8. * 2. * PI / 5.),
                                         std::sin(8. * 2. * PI / 5.)},

      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(3. * 2. * PI / 5.),
                                         std::sin(3. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(6. * 2. * PI / 5.),
                                         std::sin(6. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(9. * 2. * PI / 5.),
                                         std::sin(9. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(12. * 2. * PI / 5.),
                                         std::sin(12. * 2. * PI / 5.)},

      COMPLEX_SQRT5_5,
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(4. * 2. * PI / 5.),
                                         std::sin(4. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(8. * 2. * PI / 5.),
                                         std::sin(8. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(12. * 2. * PI / 5.),
                                         std::sin(12. * 2. * PI / 5.)},
      COMPLEX_SQRT5_5 * dd::ComplexValue{std::cos(16. * 2. * PI / 5.),
                                         std::sin(16. * 2. * PI / 5.)}};
}

inline QuintMatrix RXY5(fp theta, fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 4 or levb > 4) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuintMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(5 * leva + leva) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  identity.at(5 * leva + levb) =
      dd::ComplexValue{-std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(5 * levb + leva) =
      dd::ComplexValue{std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(5 * levb + levb) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  return identity;
}

inline QuintMatrix RH5(size_t leva, size_t levb) {
  if (leva > levb or leva > 4 or levb > 4) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuintMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(5 * leva + leva) = COMPLEX_ISQRT2_2;
  identity.at(5 * leva + levb) = COMPLEX_ISQRT2_2;
  identity.at(5 * levb + leva) = COMPLEX_ISQRT2_2;
  identity.at(5 * levb + levb) = COMPLEX_MISQRT2_2;
  return identity;
}

constexpr QuintMatrix X5dag{
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
};

constexpr QuintMatrix X5{
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO};

inline QuintMatrix RZ5(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 4 or levb > 4) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuintMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(5 * leva + leva) =
      dd::ComplexValue{std::cos(phi / 2), -std::sin(phi / 2)};
  identity.at(5 * levb + levb) =
      dd::ComplexValue{std::cos(phi / 2), +std::sin(phi / 2)};
  return identity;
}
inline QuintMatrix VirtRZ5(fp phi, size_t i) {
  QuintMatrix zero = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  zero.at(i + i * 5) = dd::ComplexValue{std::cos(phi), -std::sin(phi)};
  return zero;
}

inline QuintMatrix Z5() {
  QuintMatrix id = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  for (int level = 0; level < 5; ++level) {
    const double angle = fmod(2.0 * level / 5, 2.0) * PI;
    id.at(level + level * 5) =
        dd::ComplexValue{std::cos(angle), std::sin(angle)};
  }

  return id;
}
inline QuintMatrix S5() {
  QuintMatrix id = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  for (int level = 0; level < 5; ++level) {
    const double omegaArg = fmod(2.0 / 5 * level * (level + 1) / 2.0, 2.0);
    const auto omega =
        dd::ComplexValue{std::cos(omegaArg * PI), std::sin(omegaArg * PI)};
    id.at(level + level * 5) = omega;
  }
  return id;
}
inline QuintMatrix embX5(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 4 or levb > 4) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  QuintMatrix identity = {
      COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};
  identity.at(5 * leva + leva) = COMPLEX_ZERO;
  identity.at(5 * leva + levb) =
      dd::ComplexValue{-std::sin(phi), -std::cos(phi)};
  identity.at(5 * levb + leva) =
      dd::ComplexValue{std::sin(phi), -std::cos(phi)};
  identity.at(5 * levb + levb) = COMPLEX_ZERO;
  return identity;
}
///////////////////////////////////////////////////////////////////////////////////////////
constexpr SextMatrix I6{COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

inline SextMatrix Pi6(size_t i) {
  SextMatrix zero = {
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO,
  };
  zero.at(i + i * 6) = COMPLEX_ONE;
  return zero;
}
inline SextMatrix H6() {
  return SextMatrix{
      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6,

      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6 *
          dd::ComplexValue{std::cos(2. * PI / 6.), std::sin(2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(2. * 2. * PI / 6.),
                                         std::sin(2. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(3. * 2. * PI / 6.),
                                         std::sin(3. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(4. * 2. * PI / 6.),
                                         std::sin(4. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(5. * 2. * PI / 6.),
                                         std::sin(5. * 2. * PI / 6.)},

      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(2. * 2. * PI / 6.),
                                         std::sin(2. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(4. * 2. * PI / 6.),
                                         std::sin(4. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(6. * 2. * PI / 6.),
                                         std::sin(6. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(8. * 2. * PI / 6.),
                                         std::sin(8. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(10. * 2. * PI / 6.),
                                         std::sin(10. * 2. * PI / 6.)},

      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(3. * 2. * PI / 6.),
                                         std::sin(3. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(6. * 2. * PI / 6.),
                                         std::sin(6. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(9. * 2. * PI / 6.),
                                         std::sin(9. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(12. * 2. * PI / 6.),
                                         std::sin(12. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(15. * 2. * PI / 6.),
                                         std::sin(15. * 2. * PI / 6.)},

      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(4. * 2. * PI / 6.),
                                         std::sin(4. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(8. * 2. * PI / 6.),
                                         std::sin(8. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(12. * 2. * PI / 6.),
                                         std::sin(12. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(16. * 2. * PI / 6.),
                                         std::sin(16. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(20. * 2. * PI / 6.),
                                         std::sin(20. * 2. * PI / 6.)},

      COMPLEX_SQRT6_6,
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(5. * 2. * PI / 6.),
                                         std::sin(5. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(10. * 2. * PI / 6.),
                                         std::sin(10. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(15. * 2. * PI / 6.),
                                         std::sin(15. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(20. * 2. * PI / 6.),
                                         std::sin(20. * 2. * PI / 6.)},
      COMPLEX_SQRT6_6 * dd::ComplexValue{std::cos(25. * 2. * PI / 6.),
                                         std::sin(25. * 2. * PI / 6.)},
  };
}

inline SextMatrix RXY6(fp theta, fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 5 or levb > 5) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SextMatrix identity = I6;
  identity.at(6 * leva + leva) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  identity.at(6 * leva + levb) =
      dd::ComplexValue{-std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(6 * levb + leva) =
      dd::ComplexValue{std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(6 * levb + levb) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  return identity;
}

inline SextMatrix RH6(size_t leva, size_t levb) {
  if (leva > levb or leva > 5 or levb > 5) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SextMatrix identity = I6;
  identity.at(6 * leva + leva) = COMPLEX_ISQRT2_2;
  identity.at(6 * leva + levb) = COMPLEX_ISQRT2_2;
  identity.at(6 * levb + leva) = COMPLEX_ISQRT2_2;
  identity.at(6 * levb + levb) = COMPLEX_MISQRT2_2;
  return identity;
}

constexpr SextMatrix X6dag{COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                           COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                           COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,

                           COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                           COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};

constexpr SextMatrix X6{COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ONE,  COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
                        COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO};

inline SextMatrix RZ6(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 5 or levb > 5) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SextMatrix identity = I6;
  identity.at(6 * leva + leva) =
      dd::ComplexValue{std::cos(phi / 2), -std::sin(phi / 2)};
  identity.at(6 * levb + levb) =
      dd::ComplexValue{std::cos(phi / 2), +std::sin(phi / 2)};
  return identity;
}
inline SextMatrix VirtRZ6(fp phi, size_t i) {
  SextMatrix identity = I6;
  identity.at(i + i * 6) = dd::ComplexValue{std::cos(phi), -std::sin(phi)};
  return identity;
}

inline SextMatrix Z6() {
  SextMatrix id = I6;
  for (int level = 0; level < 6; ++level) {
    const double angle = fmod(2.0 * level / 6, 2.0) * PI;
    id.at(level + level * 6) =
        dd::ComplexValue{std::cos(angle), std::sin(angle)};
  }

  return id;
}
inline SextMatrix S6() {
  SextMatrix id = I6;
  for (int level = 0; level < 6; ++level) {
    const double omegaArg = fmod(2.0 / 6 * level * (level + 1) / 2.0, 2.0);
    const auto omega =
        dd::ComplexValue{std::cos(omegaArg * PI), std::sin(omegaArg * PI)};
    id.at(level + level * 6) = omega;
  }
  return id;
}
inline SextMatrix embX6(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 5 or levb > 5) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SextMatrix identity = I6;
  identity.at(6 * leva + leva) = COMPLEX_ZERO;
  identity.at(6 * leva + levb) =
      dd::ComplexValue{-std::sin(phi), -std::cos(phi)};
  identity.at(6 * levb + leva) =
      dd::ComplexValue{std::sin(phi), -std::cos(phi)};
  identity.at(6 * levb + levb) = COMPLEX_ZERO;
  return identity;
}
///////////////////////////////////////////////////////////////////////////////////////////
constexpr SeptMatrix I7{
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE};

inline SeptMatrix Pi7(size_t i) {
  SeptMatrix zero = {
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
      COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};
  zero.at(i + i * 7) = COMPLEX_ONE;
  return zero;
}
inline SeptMatrix H7() {
  return SeptMatrix{
      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7,

      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7 *
          dd::ComplexValue{std::cos(2. * PI / 7.), std::sin(2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(2. * 2. * PI / 7.),
                                         std::sin(2. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(3. * 2. * PI / 7.),
                                         std::sin(3. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(4. * 2. * PI / 7.),
                                         std::sin(4. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(5. * 2. * PI / 7.),
                                         std::sin(5. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(6. * 2. * PI / 7.),
                                         std::sin(6. * 2. * PI / 7.)},

      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(2. * 2. * PI / 7.),
                                         std::sin(2. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(4. * 2. * PI / 7.),
                                         std::sin(4. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(6. * 2. * PI / 7.),
                                         std::sin(6. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(8. * 2. * PI / 7.),
                                         std::sin(8. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(10. * 2. * PI / 7.),
                                         std::sin(10. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(12. * 2. * PI / 7.),
                                         std::sin(12. * 2. * PI / 7.)},

      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(3. * 2. * PI / 7.),
                                         std::sin(3. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(6. * 2. * PI / 7.),
                                         std::sin(6. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(9. * 2. * PI / 7.),
                                         std::sin(9. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(12. * 2. * PI / 7.),
                                         std::sin(12. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(15. * 2. * PI / 7.),
                                         std::sin(15. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(18. * 2. * PI / 7.),
                                         std::sin(18. * 2. * PI / 7.)},

      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(4. * 2. * PI / 7.),
                                         std::sin(4. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(8. * 2. * PI / 7.),
                                         std::sin(8. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(12. * 2. * PI / 7.),
                                         std::sin(12. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(16. * 2. * PI / 7.),
                                         std::sin(16. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(20. * 2. * PI / 7.),
                                         std::sin(20. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(24. * 2. * PI / 7.),
                                         std::sin(24. * 2. * PI / 7.)},

      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(5. * 2. * PI / 7.),
                                         std::sin(5. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(10. * 2. * PI / 7.),
                                         std::sin(10. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(15. * 2. * PI / 7.),
                                         std::sin(15. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(20. * 2. * PI / 7.),
                                         std::sin(20. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(25. * 2. * PI / 7.),
                                         std::sin(25. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(30. * 2. * PI / 7.),
                                         std::sin(30. * 2. * PI / 7.)},

      COMPLEX_SQRT7_7,
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(6. * 2. * PI / 7.),
                                         std::sin(6. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(12. * 2. * PI / 7.),
                                         std::sin(12. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(18. * 2. * PI / 7.),
                                         std::sin(18. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(24. * 2. * PI / 7.),
                                         std::sin(24. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(30. * 2. * PI / 7.),
                                         std::sin(30. * 2. * PI / 7.)},
      COMPLEX_SQRT7_7 * dd::ComplexValue{std::cos(36. * 2. * PI / 7.),
                                         std::sin(36. * 2. * PI / 7.)}};
}

inline SeptMatrix RXY7(fp theta, fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 6 or levb > 6) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SeptMatrix identity = I7;
  identity.at(7 * leva + leva) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  identity.at(7 * leva + levb) =
      dd::ComplexValue{-std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(7 * levb + leva) =
      dd::ComplexValue{std::sin(theta / 2.) * std::sin(phi),
                       -std::sin(theta / 2.) * std::cos(phi)};
  identity.at(7 * levb + levb) = dd::ComplexValue{std::cos(theta / 2.), 0.};
  return identity;
}

inline SeptMatrix RH7(size_t leva, size_t levb) {
  if (leva > levb or leva > 6 or levb > 6) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SeptMatrix identity = I7;
  identity.at(7 * leva + leva) = COMPLEX_ISQRT2_2;
  identity.at(7 * leva + levb) = COMPLEX_ISQRT2_2;
  identity.at(7 * levb + leva) = COMPLEX_ISQRT2_2;
  identity.at(7 * levb + levb) = COMPLEX_MISQRT2_2;
  return identity;
}

constexpr SeptMatrix X7dag{
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,

    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,

    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,

    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,

    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO};

constexpr SeptMatrix X7{
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ZERO,
    COMPLEX_ZERO, COMPLEX_ZERO, COMPLEX_ONE,  COMPLEX_ZERO};

inline SeptMatrix RZ7(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 6 or levb > 6) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SeptMatrix identity = I7;
  identity.at(7 * leva + leva) =
      dd::ComplexValue{std::cos(phi / 2), -std::sin(phi / 2)};
  identity.at(7 * levb + levb) =
      dd::ComplexValue{std::cos(phi / 2), +std::sin(phi / 2)};
  return identity;
}
inline SeptMatrix VirtRZ7(fp phi, size_t i) {
  SeptMatrix identity = I7;
  identity.at(i + i * 7) = dd::ComplexValue{std::cos(phi), -std::sin(phi)};
  return identity;
}

inline SeptMatrix Z7() {
  SeptMatrix id = I7;
  for (int level = 0; level < 7; ++level) {
    const double angle = fmod(2.0 * level / 7, 2.0) * PI;
    id.at(level + level * 7) =
        dd::ComplexValue{std::cos(angle), std::sin(angle)};
  }

  return id;
}
inline SeptMatrix S7() {
  SeptMatrix id = I7;
  for (int level = 0; level < 7; ++level) {
    const double omegaArg = fmod(2.0 / 7 * level * (level + 1) / 2.0, 2.0);
    const auto omega =
        dd::ComplexValue{std::cos(omegaArg * PI), std::sin(omegaArg * PI)};
    id.at(level + level * 7) = omega;
  }
  return id;
}
inline SeptMatrix embX7(fp phi, size_t leva, size_t levb) {
  if (leva > levb or leva > 6 or levb > 6) {
    throw std::invalid_argument("LEV A cannot be higher than  LEV B");
  }
  SeptMatrix identity = I7;
  identity.at(7 * leva + leva) = COMPLEX_ZERO;
  identity.at(7 * leva + levb) =
      dd::ComplexValue{-std::sin(phi), -std::cos(phi)};
  identity.at(7 * levb + leva) =
      dd::ComplexValue{std::sin(phi), -std::cos(phi)};
  identity.at(7 * levb + levb) = COMPLEX_ZERO;
  return identity;
}

// NOLINTEND(readability-identifier-naming)
} // namespace dd
#endif // DD_PACKAGE_GATEMATRIXDEFINITIONS_H
