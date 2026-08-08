#pragma once

#include "dtdma/shared_tridiagonal_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dtdma {

namespace detail {

inline std::size_t storage_system_count(
    const SharedTridiagonalBatch&) noexcept {
  return 1;
}
inline std::size_t storage_system_count(
    const TridiagonalBatch& batch) noexcept {
  return batch.batch_size();
}
inline std::size_t first_rhs_system(const SharedTridiagonalBatch&,
                                    const std::size_t) noexcept {
  return 0;
}
inline std::size_t last_rhs_system(const SharedTridiagonalBatch& batch,
                                   const std::size_t) noexcept {
  return batch.batch_size();
}
inline std::size_t first_rhs_system(const TridiagonalBatch&,
                                    const std::size_t storage_system) {
  return storage_system;
}
inline std::size_t last_rhs_system(const TridiagonalBatch&,
                                   const std::size_t storage_system) {
  return storage_system + 1;
}

}  // namespace detail

template <typename TridiagonalStorage>
inline void batched_thomas_solve(TridiagonalStorage& batch) {
  const std::size_t row_count = batch.row_count();
  if (row_count < 2) {
    throw std::invalid_argument(
        "a tridiagonal system must have at least two rows");
  }
  const TridiagonalStorage& input = batch;

  std::vector<Scalar> lower(row_count);
  std::vector<Scalar> diagonal(row_count);
  std::vector<Scalar> upper(row_count);

  for (std::size_t storage_system = 0;
       storage_system < detail::storage_system_count(input);
       ++storage_system) {
    for (std::size_t row = 0; row < row_count; ++row) {
      lower[row] = input.lower(row, storage_system);
      diagonal[row] = input.diagonal(row, storage_system);
      upper[row] = input.upper(row, storage_system);
    }

    const std::size_t first_system =
        detail::first_rhs_system(input, storage_system);
    const std::size_t last_system =
        detail::last_rhs_system(input, storage_system);

    for (std::size_t row = 1; row < row_count; ++row) {
      const Scalar multiplier = lower[row] / diagonal[row - 1];
      diagonal[row] = diagonal[row] - multiplier * upper[row - 1];
      for (std::size_t system = first_system; system < last_system;
           ++system) {
        batch.rhs(row, system) =
            batch.rhs(row, system) -
            multiplier * batch.rhs(row - 1, system);
      }
    }

    for (std::size_t system = first_system; system < last_system;
         ++system) {
      batch.rhs(row_count - 1, system) =
          batch.rhs(row_count - 1, system) / diagonal[row_count - 1];
    }

    for (std::size_t row = row_count - 1; row-- > 0;) {
      for (std::size_t system = first_system; system < last_system;
           ++system) {
        batch.rhs(row, system) =
            (batch.rhs(row, system) -
             upper[row] * batch.rhs(row + 1, system)) /
            diagonal[row];
      }
    }
  }
}

}  // namespace dtdma
