#pragma once

#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/backward_rhs_reduction.hpp"
#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"
#include "dtdma/single_partition_reconstruction.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/virtual_partitioning.hpp"
#include "dtdma/virtual_reduced_system.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dtdma {

inline void solve_virtual_system(
    const TridiagonalBatch& original,
    const VirtualPartitioning& partitioning,
    ReducedRhsBatch& solution) {
  if (partitioning.global_row_count() != original.row_count() ||
      solution.row_count() != original.row_count() ||
      solution.batch_size() != original.batch_size()) {
    throw std::invalid_argument(
        "global operator, partitioning, and solution dimensions must match");
  }

  const std::size_t partition_count = partitioning.partition_count();
  const std::size_t batch_size = original.batch_size();

  std::vector<PreparedOperatorBatch> prepared_partitions;
  std::vector<ReducedRhsBatch> working_partitions;
  std::vector<ReducedRhsEndpoints> partition_endpoints;
  prepared_partitions.reserve(partition_count);
  working_partitions.reserve(partition_count);
  partition_endpoints.reserve(partition_count);

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    const auto partition = partitioning.partition(rank);
    const std::size_t local_row_count = partition.local_row_count;
    TridiagonalBatch local_original(local_row_count, batch_size);

    for (std::size_t local_row = 0;
         local_row < local_row_count; ++local_row) {
      const std::size_t global_row = partition.begin_row + local_row;
      for (std::size_t system = 0; system < batch_size; ++system) {
        local_original.lower(local_row, system) =
            original.lower(global_row, system);
        local_original.diagonal(local_row, system) =
            original.diagonal(global_row, system);
        local_original.upper(local_row, system) =
            original.upper(global_row, system);
        local_original.rhs(local_row, system) =
            original.rhs(global_row, system);
      }
    }

    prepared_partitions.emplace_back(local_row_count, batch_size);
    working_partitions.emplace_back(local_row_count, batch_size);
    partition_endpoints.emplace_back(batch_size);

    prepare_forward_coefficients(local_original,
                                 prepared_partitions.back());
    prepare_backward_coefficients(local_original,
                                  prepared_partitions.back());
    initialize_reduced_rhs(local_original, working_partitions.back());
    reduce_rhs_forward(local_original, prepared_partitions.back(),
                       working_partitions.back());
    reduce_rhs_backward(local_original, prepared_partitions.back(),
                        working_partitions.back());
    extract_reduced_rhs_endpoints(working_partitions.back(),
                                  partition_endpoints.back());
  }

  TridiagonalBatch reduced_system(2 * partition_count, batch_size);
  assemble_virtual_reduced_system(
      partitioning, prepared_partitions, partition_endpoints, reduced_system);
  batched_thomas_solve(reduced_system);
  recover_virtual_reduced_endpoints(partitioning, reduced_system,
                                    partition_endpoints);

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    reconstruct_single_partition(prepared_partitions[rank],
                                 working_partitions[rank],
                                 partition_endpoints[rank]);
    const auto partition = partitioning.partition(rank);
    const std::size_t local_row_count = partition.local_row_count;
    for (std::size_t local_row = 0;
         local_row < local_row_count; ++local_row) {
      const std::size_t global_row = partition.begin_row + local_row;
      for (std::size_t system = 0; system < batch_size; ++system) {
        solution.rhs(global_row, system) =
            working_partitions[rank].rhs(local_row, system);
      }
    }
  }
}

}  // namespace dtdma
