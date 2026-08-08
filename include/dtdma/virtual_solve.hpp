#pragma once

#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/backward_rhs_reduction.hpp"
#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"
#include "dtdma/shared_prepared_operator.hpp"
#include "dtdma/shared_tridiagonal_batch.hpp"
#include "dtdma/shifted_diagonal_tridiagonal_batch.hpp"
#include "dtdma/single_partition_reconstruction.hpp"
#include "dtdma/system_diagonal_tridiagonal_batch.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/virtual_partitioning.hpp"
#include "dtdma/virtual_reduced_system.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

namespace detail {

inline SharedTridiagonalBatch make_local_operator(
    const SharedTridiagonalBatch& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  SharedTridiagonalBatch local(local_row_count, original.batch_size());
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    local.lower(local_row) = original.lower(global_row);
    local.diagonal(local_row) = original.diagonal(global_row);
    local.upper(local_row) = original.upper(global_row);
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      local.rhs(local_row, system) = original.rhs(global_row, system);
    }
  }
  return local;
}

inline ShiftedDiagonalTridiagonalBatch make_local_operator(
    const ShiftedDiagonalTridiagonalBatch& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  ShiftedDiagonalTridiagonalBatch local(local_row_count,
                                        original.batch_size());
  for (std::size_t system = 0; system < original.batch_size(); ++system) {
    local.shift(system) = original.shift(system);
  }
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    local.lower(local_row) = original.lower(global_row);
    local.base_diagonal(local_row) = original.base_diagonal(global_row);
    local.upper(local_row) = original.upper(global_row);
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      local.rhs(local_row, system) = original.rhs(global_row, system);
    }
  }
  return local;
}

inline SystemDiagonalTridiagonalBatch make_local_operator(
    const SystemDiagonalTridiagonalBatch& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  SystemDiagonalTridiagonalBatch local(local_row_count,
                                       original.batch_size());
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    local.lower(local_row) = original.lower(global_row);
    local.upper(local_row) = original.upper(global_row);
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      local.diagonal(local_row, system) =
          original.diagonal(global_row, system);
      local.rhs(local_row, system) = original.rhs(global_row, system);
    }
  }
  return local;
}

inline TridiagonalBatch make_local_operator(
    const TridiagonalBatch& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  TridiagonalBatch local(local_row_count, original.batch_size());
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      local.lower(local_row, system) = original.lower(global_row, system);
      local.diagonal(local_row, system) =
          original.diagonal(global_row, system);
      local.upper(local_row, system) = original.upper(global_row, system);
      local.rhs(local_row, system) = original.rhs(global_row, system);
    }
  }
  return local;
}

}  // namespace detail

template <typename OriginalOperator>
inline void solve_virtual_system(
    const OriginalOperator& original,
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
  using PreparedOperator = typename OriginalOperator::PreparedOperator;
  using ReducedOperator = typename OriginalOperator::ReducedOperator;

  std::vector<PreparedOperator> prepared_partitions;
  std::vector<ReducedRhsBatch> working_partitions;
  std::vector<ReducedRhsEndpoints> partition_endpoints;
  prepared_partitions.reserve(partition_count);
  working_partitions.reserve(partition_count);
  partition_endpoints.reserve(partition_count);

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    const auto partition = partitioning.partition(rank);
    const std::size_t local_row_count = partition.local_row_count;
    auto local_original = detail::make_local_operator(
        original, partition.begin_row, local_row_count);

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

  ReducedOperator reduced_system(2 * partition_count, batch_size);
  assemble_virtual_reduced_system(
      partitioning,
      std::span<const PreparedOperator>(prepared_partitions),
      partition_endpoints, reduced_system);
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
