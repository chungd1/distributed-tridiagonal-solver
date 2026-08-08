#pragma once

#include "dtdma/detail/coefficient_dimensions.hpp"
#include "dtdma/rhs_batch.hpp"
#include "dtdma/tridiagonal_shared.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dtdma {

namespace detail {

inline std::size_t first_rhs_system(const TridiagonalShared&,
                                    const RhsBatch&,
                                    const std::size_t) noexcept {
  return 0;
}
inline std::size_t last_rhs_system(const TridiagonalShared&,
                                   const RhsBatch& rhs,
                                   const std::size_t) noexcept {
  return rhs.batch_size();
}
inline std::size_t first_rhs_system(const TridiagonalBatch&,
                                    const RhsBatch&,
                                    const std::size_t storage_system) {
  return storage_system;
}
inline std::size_t last_rhs_system(const TridiagonalBatch&,
                                   const RhsBatch&,
                                   const std::size_t storage_system) {
  return storage_system + 1;
}

}  // namespace detail

template <typename TridiagonalStorage>
inline void batched_thomas_solve(const TridiagonalStorage& coefficients,
                                 RhsBatch& rhs) {
  const std::size_t row_count = coefficients.row_count();
  if (row_count < 2) {
    throw std::invalid_argument(
        "a tridiagonal system must have at least two rows");
  }
  if (rhs.row_count() != row_count ||
      !detail::rhs_batch_size_is_compatible(coefficients,
                                             rhs.batch_size())) {
    throw std::invalid_argument(
        "coefficient and RHS batch dimensions must match");
  }

  std::vector<Scalar> lower(row_count);
  std::vector<Scalar> diagonal(row_count);
  std::vector<Scalar> upper(row_count);

  for (std::size_t storage_system = 0;
       storage_system < detail::storage_system_count(coefficients);
       ++storage_system) {
    for (std::size_t row = 0; row < row_count; ++row) {
      lower[row] = coefficients.lower(row, storage_system);
      diagonal[row] = coefficients.diagonal(row, storage_system);
      upper[row] = coefficients.upper(row, storage_system);
    }

    const std::size_t first_system =
        detail::first_rhs_system(coefficients, rhs, storage_system);
    const std::size_t last_system =
        detail::last_rhs_system(coefficients, rhs, storage_system);

    for (std::size_t row = 1; row < row_count; ++row) {
      const Scalar multiplier = lower[row] / diagonal[row - 1];
      diagonal[row] = diagonal[row] - multiplier * upper[row - 1];
      for (std::size_t system = first_system; system < last_system;
           ++system) {
        rhs.rhs(row, system) =
            rhs.rhs(row, system) -
            multiplier * rhs.rhs(row - 1, system);
      }
    }

    for (std::size_t system = first_system; system < last_system;
         ++system) {
      rhs.rhs(row_count - 1, system) =
          rhs.rhs(row_count - 1, system) / diagonal[row_count - 1];
    }

    for (std::size_t row = row_count - 1; row-- > 0;) {
      for (std::size_t system = first_system; system < last_system;
           ++system) {
        rhs.rhs(row, system) =
            (rhs.rhs(row, system) -
             upper[row] * rhs.rhs(row + 1, system)) /
            diagonal[row];
      }
    }
  }
}

}  // namespace dtdma
