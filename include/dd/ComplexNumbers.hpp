/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DDcomplex_H
#define DDcomplex_H

#include "Complex.hpp"
#include "ComplexCache.hpp"
#include "ComplexTable.hpp"
#include "ComplexValue.hpp"
#include "Definitions.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>

namespace dd {
struct ComplexNumbers {
  ComplexTable<> complexTable{};
  ComplexCache<> complexCache{};

  ComplexNumbers() = default;
  ~ComplexNumbers() = default;

  void clear() {
    complexTable.clear();
    complexCache.clear();
  }

  static void setTolerance(fp tol) { ComplexTable<>::setTolerance(tol); }

  // operations on complex numbers
  // meanings are self-evident from the names
  static void add(Complex& r, const Complex& a, const Complex& b) {
    assert(r != Complex::zero);
    assert(r != Complex::one);
    r.real->value = CTEntry::val(a.real) + CTEntry::val(b.real);
    r.img->value = CTEntry::val(a.img) + CTEntry::val(b.img);
  }
  static void sub(Complex& r, const Complex& a, const Complex& b) {
    assert(r != Complex::zero);
    assert(r != Complex::one);
    r.real->value = CTEntry::val(a.real) - CTEntry::val(b.real);
    r.img->value = CTEntry::val(a.img) - CTEntry::val(b.img);
  }
  static void mul(Complex& r, const Complex& a, const Complex& b) {
    assert(r != Complex::zero);
    assert(r != Complex::one);
    if (a.approximatelyOne()) {
      r.setVal(b);
    } else if (b.approximatelyOne()) {
      r.setVal(a);
    } else if (a.approximatelyZero() || b.approximatelyZero()) {
      r.real->value = 0.;
      r.img->value = 0.;
    } else {
      const auto ar = CTEntry::val(a.real);
      const auto ai = CTEntry::val(a.img);
      const auto br = CTEntry::val(b.real);
      const auto bi = CTEntry::val(b.img);

      r.real->value = ar * br - ai * bi;
      r.img->value = ar * bi + ai * br;
    }
  }
  static void div(Complex& r, const Complex& a, const Complex& b) {
    assert(r != Complex::zero);
    assert(r != Complex::one);
    if (a.approximatelyEquals(b)) {
      r.real->value = 1.;
      r.img->value = 0.;
    } else if (b.approximatelyOne()) {
      r.setVal(a);
    } else {
      const auto ar = CTEntry::val(a.real);
      const auto ai = CTEntry::val(a.img);
      const auto br = CTEntry::val(b.real);
      const auto bi = CTEntry::val(b.img);

      const auto cmag = br * br + bi * bi;

      r.real->value = (ar * br + ai * bi) / cmag;
      r.img->value = (ai * br - ar * bi) / cmag;
    }
  }
  static inline fp mag2(const Complex& a) {
    auto ar = CTEntry::val(a.real);
    auto ai = CTEntry::val(a.img);

    return ar * ar + ai * ai;
  }
  static inline fp mag(const Complex& a) { return std::sqrt(mag2(a)); }
  static inline fp arg(const Complex& a) {
    auto ar = CTEntry::val(a.real);
    auto ai = CTEntry::val(a.img);
    return std::atan2(ai, ar);
  }
  static Complex conj(const Complex& a) {
    auto ret = a;
    if (a.img != Complex::zero.img) {
      ret.img = CTEntry::flipPointerSign(a.img);
    }
    return ret;
  }
  static Complex neg(const Complex& a) {
    auto ret = a;
    if (a.img != Complex::zero.img) {
      ret.img = CTEntry::flipPointerSign(a.img);
    }
    if (a.real != Complex::zero.img) {
      ret.real = CTEntry::flipPointerSign(a.real);
    }
    return ret;
  }

  inline Complex addCached(const Complex& a, const Complex& b) {
    auto c = getCached();
    add(c, a, b);
    return c;
  }

  inline Complex subCached(const Complex& a, const Complex& b) {
    auto c = getCached();
    sub(c, a, b);
    return c;
  }

  inline Complex mulCached(const Complex& a, const Complex& b) {
    auto c = getCached();
    mul(c, a, b);
    return c;
  }

  inline Complex divCached(const Complex& a, const Complex& b) {
    auto c = getCached();
    div(c, a, b);
    return c;
  }

  // lookup a complex value in the complex table; if not found add it
  Complex lookup(const Complex& c) {
    if (c == Complex::zero) {
      return Complex::zero;
    }
    if (c == Complex::one) {
      return Complex::one;
    }

    auto valr = CTEntry::val(c.real);
    auto vali = CTEntry::val(c.img);
    return lookup(valr, vali);
  }
  Complex lookup(const fp& r, const fp& i) {
    Complex ret{};

    const auto signR = std::signbit(r);
    if (signR) {
      const auto absr = std::abs(r);
      // if absolute value is close enough to zero, just return the zero entry
      // (avoiding -0.0)
      if (absr < decltype(complexTable)::tolerance()) {
        ret.real = &decltype(complexTable)::zero;
      } else {
        ret.real = CTEntry::getNegativePointer(complexTable.lookup(absr));
      }
    } else {
      ret.real = complexTable.lookup(r);
    }

    const auto signI = std::signbit(i);
    if (signI) {
      const auto absi = std::abs(i);
      // if absolute value is close enough to zero, just return the zero entry
      // (avoiding -0.0)
      if (absi < decltype(complexTable)::tolerance()) {
        ret.img = &decltype(complexTable)::zero;
      } else {
        ret.img = CTEntry::getNegativePointer(complexTable.lookup(absi));
      }
    } else {
      ret.img = complexTable.lookup(i);
    }

    return ret;
  }
  inline Complex lookup(const ComplexValue& c) { return lookup(c.r, c.i); }

  // reference counting and garbage collection
  static void incRef(const Complex& c) {
    // `zero` and `one` are static and never altered
    if (c != Complex::zero && c != Complex::one) {
      ComplexTable<>::incRef(c.real);
      ComplexTable<>::incRef(c.img);
    }
  }
  static void decRef(const Complex& c) {
    // `zero` and `one` are static and never altered
    if (c != Complex::zero && c != Complex::one) {
      ComplexTable<>::decRef(c.real);
      ComplexTable<>::decRef(c.img);
    }
  }
  std::size_t garbageCollect(bool force = false) {
    return complexTable.garbageCollect(force);
  }

  // provide (temporary) cached complex number
  inline Complex getTemporary() { return complexCache.getTemporaryComplex(); }

  inline Complex getTemporary(const fp& r, const fp& i) {
    auto c = complexCache.getTemporaryComplex();
    c.real->value = r;
    c.img->value = i;
    return c;
  }

  inline Complex getTemporary(const ComplexValue& c) {
    return getTemporary(c.r, c.i);
  }

  inline Complex getCached() { return complexCache.getCachedComplex(); }

  inline Complex getCached(const fp& r, const fp& i) {
    auto c = complexCache.getCachedComplex();
    c.real->value = r;
    c.img->value = i;
    return c;
  }

  inline Complex getCached(const ComplexValue& c) {
    return getCached(c.r, c.i);
  }

  void returnToCache(Complex& c) { complexCache.returnToCache(c); }

  [[nodiscard]] std::size_t cacheCount() const {
    return complexCache.getCount();
  }
};
} // namespace dd
#endif
