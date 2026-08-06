#pragma once

#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

inline void reconstruct_single_partition(
    const PreparedOperatorBatch& prepared,
    ReducedRhsBatch& working,
    const ReducedRhsEndpoints& solved_endpoints) {
  if (prepared.row_count() < 3) {
    throw std::invalid_argument(
        "single-partition reconstruction requires at least three rows");
  }
  if (working.row_count() != prepared.row_count() ||
      working.batch_size() != prepared.batch_size() ||
      solved_endpoints.batch_size() != prepared.batch_size()) {
    throw std::invalid_argument(
        "prepared, RHS working, and endpoint dimensions must match");
  }

  const std::size_t last_row = prepared.row_count() - 1;
  for (std::size_t system = 0; system < prepared.batch_size(); ++system) {
    const Scalar first_endpoint = solved_endpoints.endpoint(system, 0);
    const Scalar last_endpoint = solved_endpoints.endpoint(system, 1);

    working.rhs(0, system) = first_endpoint;
    working.rhs(last_row, system) = last_endpoint;

    for (std::size_t row = 1; row < last_row; ++row) {
      working.rhs(row, system) =
          (working.rhs(row, system) -
           prepared.prepared_lower(row, system) * first_endpoint -
           prepared.prepared_upper(row, system) * last_endpoint) /
          prepared.prepared_diagonal(row, system);
    }
  }
}

}  // namespace dtdma
