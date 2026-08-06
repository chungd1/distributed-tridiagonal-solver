#pragma once

#include "dtdma/indexing.hpp"
#include "dtdma/thomas_solver.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <vector>

namespace dtdma {

inline void batched_thomas_solve(TridiagonalBatch& batch) {
  const std::size_t row_count = batch.row_count();
  const std::size_t batch_size = batch.batch_size();
  const TridiagonalBatch& input = batch;

  std::vector<Scalar> lower(row_count);
  std::vector<Scalar> diagonal(row_count);
  std::vector<Scalar> upper(row_count);
  std::vector<Scalar> rhs(row_count);

  for (std::size_t system = 0; system < batch_size; ++system) {
    for (std::size_t row = 0; row < row_count; ++row) {
      const std::size_t index = canonical_index(row, system, batch_size);
      lower[row] = input.lower()[index];
      diagonal[row] = input.diagonal()[index];
      upper[row] = input.upper()[index];
      rhs[row] = input.rhs()[index];
    }

    thomas_solve(lower, diagonal, upper, rhs);

    for (std::size_t row = 0; row < row_count; ++row) {
      const std::size_t index = canonical_index(row, system, batch_size);
      batch.rhs()[index] = rhs[row];
    }
  }
}

}  // namespace dtdma
