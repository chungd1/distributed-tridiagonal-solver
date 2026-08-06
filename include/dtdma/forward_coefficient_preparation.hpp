#pragma once

#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

inline void prepare_forward_coefficients(
    const TridiagonalBatch& original,
    PreparedOperatorBatch& prepared) {
  if (original.row_count() < 3) {
    throw std::invalid_argument(
        "coefficient preparation requires at least three rows");
  }
  if (prepared.row_count() != original.row_count() ||
      prepared.batch_size() != original.batch_size()) {
    throw std::invalid_argument(
        "original and prepared batch dimensions must match");
  }

  for (std::size_t system = 0; system < original.batch_size(); ++system) {
    prepared.prepared_lower(1, system) = original.lower(1, system);
    prepared.prepared_diagonal(1, system) = original.diagonal(1, system);

    for (std::size_t row = 2; row < original.row_count(); ++row) {
      prepared.prepared_diagonal(row, system) =
          original.diagonal(row, system) -
          original.lower(row, system) * original.upper(row - 1, system) /
              prepared.prepared_diagonal(row - 1, system);

      prepared.prepared_lower(row, system) =
          -original.lower(row, system) *
          prepared.prepared_lower(row - 1, system) /
          prepared.prepared_diagonal(row - 1, system);
    }
  }
}

}  // namespace dtdma
