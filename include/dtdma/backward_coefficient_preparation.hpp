#pragma once

#include "dtdma/detail/coefficient_dimensions.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

template <typename OriginalOperator, typename PreparedOperator>
inline void prepare_backward_coefficients(const OriginalOperator& original,
                                          PreparedOperator& prepared) {
  if (original.row_count() < 3) {
    throw std::invalid_argument(
        "coefficient preparation requires at least three rows");
  }
  if (prepared.row_count() != original.row_count() ||
      prepared.storage_system_count() !=
          detail::storage_system_count(original)) {
    throw std::invalid_argument(
        "original and prepared batch dimensions must match");
  }

  const std::size_t last_row = original.row_count() - 1;

  for (std::size_t storage_system = 0;
       storage_system < prepared.storage_system_count(); ++storage_system) {
    prepared.prepared_upper(last_row, storage_system) =
        original.upper(last_row, storage_system);
    prepared.prepared_upper(last_row - 1, storage_system) =
        original.upper(last_row - 1, storage_system);

    for (std::size_t row = last_row - 2; row > 0; --row) {
      prepared.prepared_lower(row, storage_system) =
          prepared.prepared_lower(row, storage_system) -
          original.upper(row, storage_system) *
              prepared.prepared_lower(row + 1, storage_system) /
              prepared.prepared_diagonal(row + 1, storage_system);

      prepared.prepared_upper(row, storage_system) =
          -original.upper(row, storage_system) *
          prepared.prepared_upper(row + 1, storage_system) /
          prepared.prepared_diagonal(row + 1, storage_system);
    }

    prepared.prepared_lower(0, storage_system) =
        original.lower(0, storage_system);
    prepared.prepared_diagonal(0, storage_system) =
        original.diagonal(0, storage_system) -
        original.upper(0, storage_system) *
            prepared.prepared_lower(1, storage_system) /
            original.diagonal(1, storage_system);
    prepared.prepared_upper(0, storage_system) =
        -original.upper(0, storage_system) *
        prepared.prepared_upper(1, storage_system) /
        original.diagonal(1, storage_system);
  }
}

}  // namespace dtdma
