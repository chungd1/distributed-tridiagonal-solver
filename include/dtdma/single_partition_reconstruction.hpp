#pragma once

#include "dtdma/detail/coefficient_dimensions.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/rhs_batch.hpp"
#include "dtdma/endpoint_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

template <typename PreparedOperator>
inline void reconstruct_single_partition(
    const PreparedOperator& prepared,
    RhsBatch& working,
    const EndpointBatch& solved_endpoints) {
  if (prepared.row_count() < 3) {
    throw std::invalid_argument(
        "single-partition reconstruction requires at least three rows");
  }
  if (working.row_count() != prepared.row_count() ||
      solved_endpoints.batch_size() != working.batch_size() ||
      !detail::rhs_batch_size_is_compatible(prepared,
                                             working.batch_size())) {
    throw std::invalid_argument(
        "prepared, RHS working, and endpoint dimensions must match");
  }

  const std::size_t last_row = prepared.row_count() - 1;
  for (std::size_t system = 0; system < working.batch_size(); ++system) {
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
