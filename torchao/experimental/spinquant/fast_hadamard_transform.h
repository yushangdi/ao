// (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "fast_hadamard_transform_special.h"

namespace torchao {
namespace detail {

// Square root of 1 << log2_n.
template <typename T>
T fast_sqrt(int log2_n) {
  // The square root of 2**N is, by definition, 2**(N/2), which is
  // trivial to compute for even N.
  //
  // For odd N, 2**(N/2) = 2**(floor(N/2) + 1/2)
  //                     = 2**(floor(N/2)) * (2 ** (1/2))
  //                     = 2**(floor(N/2)) * sqrt(2)
  // which is again fast to compute.
  return T(1 << (log2_n / 2)) * ((log2_n % 2) ? T(std::sqrt(2)) : T(1));
}

template <typename T>
void normalize_after_fht(
    T* out,
    int log2_vec_size) {
  const T inv_sqrt = T(1) / fast_sqrt<T>(log2_vec_size);
  const int vec_size = 1 << log2_vec_size;
  for (int ii = 0; ii < vec_size; ++ii) {
    out[ii] *= inv_sqrt;
  }
}

template <typename T>
void fast_hadamard_transform_simple_impl(
    T* vec,
    int log2_vec_size) {
  if (log2_vec_size == 0) {
    return;
  }

  int step = 1;
  const auto vec_size = 1 << log2_vec_size;
  while (step < vec_size) {
    for (int ii = 0; ii < vec_size; ii += step * 2) {
      for (int jj = ii; jj < ii + step; ++jj) {
        auto x = vec[jj];
        auto y = vec[jj + step];
        vec[jj] = x + y;
        vec[jj + step] = x - y;
      }
    }
    step *= 2;
  }

  normalize_after_fht(vec, log2_vec_size);
}

} // namespace detail

// Compute the fast Walsh-Hadamard transform
// (https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
// of vec, which must be of length (1 << log2_vec_size).
template <typename T>
void fast_hadamard_transform(T* vec, int log2_vec_size) {
    detail::fast_hadamard_transform_simple_impl(vec, log2_vec_size);
}

// Like fast_hadamard_transform, but vec must be of length 28 * (1 <<
// log2_vec_size) and the transform is computed by interpreting vec as
// a (28, 1 << log2_vec_size) matrix and performing 28 FHTs, followed
// by (1 << log2_vec_size) multiplications by a particular Hadamard
// matrix of size 28x28 (see special_hadamard_code_gen.py for the
// exact matrix).
template <typename T>
void fast_hadamard_transform_28N(T* vec, int log2_vec_size) {
  const int vec_size = (1 << log2_vec_size);
  for (int ii = 0; ii < 28; ++ii) {
    fast_hadamard_transform(&vec[ii * vec_size], log2_vec_size);
  }
  for (int ii = 0; ii < vec_size; ++ii) {
    hadamard_mult_28_strided(&vec[ii], vec_size);
  }
}

} // namespace torchao
