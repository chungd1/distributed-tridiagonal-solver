#pragma once

#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/uniform_virtual_partitioning.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace dtdma {

inline void assemble_uniform_reduced_system(
    const UniformVirtualPartitioning& partitioning,
    const std::span<const PreparedOperatorBatch> prepared_partitions,
    const std::span<const ReducedRhsEndpoints> reduced_rhs_endpoints,
    TridiagonalBatch& reduced_system) {
  const std::size_t partition_count = partitioning.virtual_rank_count();
  if (prepared_partitions.size() != partition_count ||
      reduced_rhs_endpoints.size() != partition_count) {
    throw std::invalid_argument(
        "one prepared and endpoint batch is required per partition");
  }
  if (reduced_system.row_count() != 2 * partition_count) {
    throw std::invalid_argument(
        "reduced-system row count must be twice the virtual rank count");
  }

  const std::size_t batch_size = reduced_system.batch_size();
  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    if (prepared_partitions[rank].row_count() !=
            partitioning.local_row_count() ||
        prepared_partitions[rank].batch_size() != batch_size ||
        reduced_rhs_endpoints[rank].batch_size() != batch_size) {
      throw std::invalid_argument(
          "partition and reduced-system dimensions must match");
    }
  }

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    const std::size_t first_reduced_row = 2 * rank;
    const std::size_t last_reduced_row = first_reduced_row + 1;
    const std::size_t last_local_row =
        prepared_partitions[rank].row_count() - 1;

    for (std::size_t system = 0; system < batch_size; ++system) {
      reduced_system.lower(first_reduced_row, system) =
          prepared_partitions[rank].prepared_lower(0, system);
      reduced_system.diagonal(first_reduced_row, system) =
          prepared_partitions[rank].prepared_diagonal(0, system);
      reduced_system.upper(first_reduced_row, system) =
          prepared_partitions[rank].prepared_upper(0, system);
      reduced_system.rhs(first_reduced_row, system) =
          reduced_rhs_endpoints[rank].endpoint(system, 0);

      reduced_system.lower(last_reduced_row, system) =
          prepared_partitions[rank].prepared_lower(last_local_row, system);
      reduced_system.diagonal(last_reduced_row, system) =
          prepared_partitions[rank].prepared_diagonal(last_local_row, system);
      reduced_system.upper(last_reduced_row, system) =
          prepared_partitions[rank].prepared_upper(last_local_row, system);
      reduced_system.rhs(last_reduced_row, system) =
          reduced_rhs_endpoints[rank].endpoint(system, 1);
    }
  }

  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::size_t final_reduced_row = 2 * partition_count - 1;
    reduced_system.lower(0, system) = 0.0F;
    reduced_system.upper(final_reduced_row, system) = 0.0F;
  }
}

inline void recover_uniform_reduced_endpoints(
    const UniformVirtualPartitioning& partitioning,
    const TridiagonalBatch& solved_reduced_system,
    const std::span<ReducedRhsEndpoints> solved_endpoints) {
  const std::size_t partition_count = partitioning.virtual_rank_count();
  if (solved_reduced_system.row_count() != 2 * partition_count ||
      solved_endpoints.size() != partition_count) {
    throw std::invalid_argument(
        "solved reduced system and endpoint partition counts must match");
  }

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    if (solved_endpoints[rank].batch_size() !=
        solved_reduced_system.batch_size()) {
      throw std::invalid_argument(
          "solved reduced-system and endpoint batch sizes must match");
    }
  }

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    for (std::size_t system = 0;
         system < solved_reduced_system.batch_size(); ++system) {
      solved_endpoints[rank].endpoint(system, 0) =
          solved_reduced_system.rhs(2 * rank, system);
      solved_endpoints[rank].endpoint(system, 1) =
          solved_reduced_system.rhs(2 * rank + 1, system);
    }
  }
}

}  // namespace dtdma
