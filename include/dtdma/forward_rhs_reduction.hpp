#pragma once

#include "dtdma/detail/coefficient_dimensions.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/rhs_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

inline void initialize_reduced_rhs(const RhsBatch& input,
                                   RhsBatch& working) {
  if (input.row_count() < 3) {
    throw std::invalid_argument("RHS reduction requires at least three rows");
  }
  if (working.row_count() != input.row_count() ||
      working.batch_size() != input.batch_size()) {
    throw std::invalid_argument(
        "input and RHS working batch dimensions must match");
  }

  for (std::size_t row = 0; row < input.row_count(); ++row) {
    for (std::size_t system = 0; system < input.batch_size(); ++system) {
      working.rhs(row, system) = input.rhs(row, system);
    }
  }
}

template <typename PreparedOperator>
inline void reduce_rhs_forward(const PreparedOperator& prepared,
                               RhsBatch& working) {
  if (prepared.row_count() < 3) {
    throw std::invalid_argument("RHS reduction requires at least three rows");
  }
  if (working.row_count() != prepared.row_count() ||
      !detail::rhs_batch_size_is_compatible(prepared,
                                             working.batch_size())) {
    throw std::invalid_argument(
        "prepared coefficient and RHS dimensions must match");
  }

  for (std::size_t system = 0; system < working.batch_size(); ++system) {
    for (std::size_t row = 2; row < prepared.row_count(); ++row) {
      working.rhs(row, system) =
          working.rhs(row, system) -
          prepared.lower(row, system) * working.rhs(row - 1, system) /
              prepared.prepared_diagonal(row - 1, system);
    }
  }
}

}  // namespace dtdma
