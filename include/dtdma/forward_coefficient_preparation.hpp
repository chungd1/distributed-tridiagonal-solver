#pragma once

#include "dtdma/detail/coefficient_dimensions.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

template <typename OriginalOperator, typename PreparedOperator>
inline void prepare_forward_coefficients(const OriginalOperator& original,
                                         PreparedOperator& prepared) {
  if (original.row_count() < 3) {
    throw std::invalid_argument(
        "coefficient preparation requires at least three rows");
  }
  if (prepared.row_count() != original.row_count() ||
      prepared.storage_system_count() !=
          detail::storage_system_count(original)) {
    throw std::invalid_argument(
        "original and prepared coefficient dimensions must match");
  }

  for (std::size_t storage_system = 0;
       storage_system < prepared.storage_system_count(); ++storage_system) {
    prepared.prepared_lower(1, storage_system) =
        original.lower(1, storage_system);
    prepared.prepared_diagonal(1, storage_system) =
        original.diagonal(1, storage_system);

    for (std::size_t row = 2; row < original.row_count(); ++row) {
      prepared.prepared_diagonal(row, storage_system) =
          original.diagonal(row, storage_system) -
          original.lower(row, storage_system) *
              original.upper(row - 1, storage_system) /
              prepared.prepared_diagonal(row - 1, storage_system);

      prepared.prepared_lower(row, storage_system) =
          -original.lower(row, storage_system) *
          prepared.prepared_lower(row - 1, storage_system) /
          prepared.prepared_diagonal(row - 1, storage_system);
    }
  }
}

}  // namespace dtdma
