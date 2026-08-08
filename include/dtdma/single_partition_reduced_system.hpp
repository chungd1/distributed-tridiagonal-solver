#pragma once

#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <stdexcept>

namespace dtdma {

template <typename OriginalOperator, typename PreparedOperator,
          typename ReducedOperator>
inline void assemble_single_partition_reduced_system(
    const OriginalOperator& original,
    const PreparedOperator& prepared,
    const ReducedRhsBatch& working,
    const ReducedRhsEndpoints& reduced_rhs_endpoints,
    ReducedOperator& reduced_system) {
  if (original.row_count() < 3) {
    throw std::invalid_argument(
        "single-partition assembly requires at least three local rows");
  }
  if (prepared.row_count() != original.row_count() ||
      prepared.batch_size() != original.batch_size() ||
      working.row_count() != original.row_count() ||
      working.batch_size() != original.batch_size() ||
      reduced_rhs_endpoints.batch_size() != original.batch_size()) {
    throw std::invalid_argument(
        "single-partition local batch dimensions must match");
  }
  if (reduced_system.row_count() != 2 ||
      reduced_system.batch_size() != original.batch_size()) {
    throw std::invalid_argument(
        "single-partition reduced system must have two matching rows");
  }

  const std::size_t last_row = original.row_count() - 1;
  for (std::size_t storage_system = 0;
       storage_system < prepared.storage_system_count(); ++storage_system) {
    reduced_system.lower(0, storage_system) = 0.0F;
    reduced_system.diagonal(0, storage_system) =
        prepared.prepared_diagonal(0, storage_system);
    reduced_system.upper(0, storage_system) =
        prepared.prepared_upper(0, storage_system);

    reduced_system.lower(1, storage_system) =
        prepared.prepared_lower(last_row, storage_system);
    reduced_system.diagonal(1, storage_system) =
        prepared.prepared_diagonal(last_row, storage_system);
    reduced_system.upper(1, storage_system) = 0.0F;
  }

  for (std::size_t system = 0; system < original.batch_size(); ++system) {
    reduced_system.rhs(0, system) =
        reduced_rhs_endpoints.endpoint(system, 0);
    reduced_system.rhs(1, system) =
        reduced_rhs_endpoints.endpoint(system, 1);
  }
}

template <typename ReducedOperator>
inline void recover_single_partition_endpoints(
    const ReducedOperator& solved_reduced_system,
    ReducedRhsEndpoints& solved_endpoints) {
  if (solved_reduced_system.row_count() != 2 ||
      solved_reduced_system.batch_size() != solved_endpoints.batch_size()) {
    throw std::invalid_argument(
        "solved reduced system must have two rows matching the endpoints");
  }

  for (std::size_t system = 0;
       system < solved_reduced_system.batch_size(); ++system) {
    solved_endpoints.endpoint(system, 0) =
        solved_reduced_system.rhs(0, system);
    solved_endpoints.endpoint(system, 1) =
        solved_reduced_system.rhs(1, system);
  }
}

}  // namespace dtdma
