/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_COMPLEX_HPP
#define DD_PACKAGE_COMPLEX_HPP

#include "ComplexTable.hpp"
#include "ComplexValue.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

namespace dd {
using CTEntry = ComplexTable<>::Entry;

struct Complex {
  CTEntry* real;
  CTEntry* img;

  static Complex
      zero; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables):
            // Making it const breaks the code
  static Complex
      one; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables):
           // Making it const breaks the code

  void setVal(const Complex& complexNum) const {
    real->value = CTEntry::val(complexNum.real);
    img->value = CTEntry::val(complexNum.img);
  }

  [[nodiscard]] inline bool
  approximatelyEquals(const Complex& complexNum) const {
    return CTEntry::approximatelyEquals(real, complexNum.real) &&
           CTEntry::approximatelyEquals(img, complexNum.img);
  };

  [[nodiscard]] inline bool approximatelyZero() const {
    return CTEntry::approximatelyZero(real) && CTEntry::approximatelyZero(img);
  }

  [[nodiscard]] inline bool approximatelyOne() const {
    return CTEntry::approximatelyOne(real) && CTEntry::approximatelyZero(img);
  }

  inline bool operator==(const Complex& other) const {
    return real == other.real && img == other.img;
  }

  inline bool operator!=(const Complex& other) const {
    return !operator==(other);
  }

  [[nodiscard]] std::string toString(bool formatted = true,
                                     int precision = -1) const {
    return ComplexValue::toString(CTEntry::val(real), CTEntry::val(img),
                                  formatted, precision);
  }

  void writeBinary(std::ostream& os) const {
    CTEntry::writeBinary(real, os);
    CTEntry::writeBinary(img, os);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Complex& complexNum) {
  return os << complexNum.toString();
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): Making it
// const breaks the code
inline Complex Complex::zero{&ComplexTable<>::zero, &ComplexTable<>::zero};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): Making it
// const breaks the code
inline Complex Complex::one{&ComplexTable<>::one, &ComplexTable<>::zero};
} // namespace dd

namespace std {
template <> struct hash<dd::Complex> {
  std::size_t operator()(dd::Complex const& complexNum) const noexcept {
    auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(complexNum.real));
    auto h2 = dd::murmur64(reinterpret_cast<std::size_t>(complexNum.img));
    return dd::combineHash(h1, h2);
  }
};
} // namespace std

#endif // DD_PACKAGE_COMPLEX_HPP
