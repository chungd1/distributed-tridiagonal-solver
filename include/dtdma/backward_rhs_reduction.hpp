#pragma once

#include "dtdma/detail/coefficient_dimensions.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/rhs_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

template <typename PreparedOperator>
inline void reduce_rhs_backward(const PreparedOperator& prepared,
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
    for (std::size_t row = prepared.row_count() - 2; row-- > 0;) {
      working.rhs(row, system) =
          working.rhs(row, system) -
          prepared.upper(row, system) * working.rhs(row + 1, system) /
              prepared.prepared_diagonal(row + 1, system);
    }
  }
}

}  // namespace dtdma
