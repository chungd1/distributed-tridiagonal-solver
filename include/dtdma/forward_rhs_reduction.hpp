#pragma once

#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

template <typename OriginalOperator>
inline void initialize_reduced_rhs(const OriginalOperator& original,
                                   ReducedRhsBatch& working) {
  if (original.row_count() < 3) {
    throw std::invalid_argument("RHS reduction requires at least three rows");
  }
  if (working.row_count() != original.row_count() ||
      working.batch_size() != original.batch_size()) {
    throw std::invalid_argument(
        "original and RHS working batch dimensions must match");
  }

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      working.rhs(row, system) = original.rhs(row, system);
    }
  }
}

template <typename OriginalOperator, typename PreparedOperator>
inline void reduce_rhs_forward(const OriginalOperator& original,
                               const PreparedOperator& prepared,
                               ReducedRhsBatch& working) {
  if (original.row_count() < 3) {
    throw std::invalid_argument("RHS reduction requires at least three rows");
  }
  if (prepared.row_count() != original.row_count() ||
      prepared.batch_size() != original.batch_size() ||
      working.row_count() != original.row_count() ||
      working.batch_size() != original.batch_size()) {
    throw std::invalid_argument(
        "original, prepared, and RHS working batch dimensions must match");
  }

  for (std::size_t system = 0; system < original.batch_size(); ++system) {
    for (std::size_t row = 2; row < original.row_count(); ++row) {
      working.rhs(row, system) =
          working.rhs(row, system) -
          original.lower(row, system) * working.rhs(row - 1, system) /
              prepared.prepared_diagonal(row - 1, system);
    }
  }
}

}  // namespace dtdma
