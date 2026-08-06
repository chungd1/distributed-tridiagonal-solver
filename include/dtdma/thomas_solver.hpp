#pragma once

#include "dtdma/scalar.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace dtdma {

inline void thomas_solve(const std::span<const Scalar> lower,
                         const std::span<Scalar> diagonal,
                         const std::span<const Scalar> upper,
                         const std::span<Scalar> rhs) {
  const std::size_t row_count = diagonal.size();

  if (row_count < 2) {
    throw std::invalid_argument(
        "a tridiagonal system must have at least two rows");
  }
  if (lower.size() != row_count || upper.size() != row_count ||
      rhs.size() != row_count) {
    throw std::invalid_argument("tridiagonal system arrays must have equal lengths");
  }

  for (std::size_t row = 1; row < row_count; ++row) {
    const Scalar multiplier = lower[row] / diagonal[row - 1];
    diagonal[row] = diagonal[row] - multiplier * upper[row - 1];
    rhs[row] = rhs[row] - multiplier * rhs[row - 1];
  }

  rhs[row_count - 1] = rhs[row_count - 1] / diagonal[row_count - 1];

  for (std::size_t row = row_count - 1; row-- > 0;) {
    rhs[row] = (rhs[row] - upper[row] * rhs[row + 1]) / diagonal[row];
  }
}

}  // namespace dtdma
