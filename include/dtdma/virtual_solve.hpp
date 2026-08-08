#pragma once

#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/backward_rhs_reduction.hpp"
#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/rhs_batch.hpp"
#include "dtdma/endpoint_batch.hpp"
#include "dtdma/shared_prepared_operator.hpp"
#include "dtdma/tridiagonal_shared.hpp"
#include "dtdma/tridiagonal_shifted_diagonal.hpp"
#include "dtdma/single_partition_reconstruction.hpp"
#include "dtdma/tridiagonal_system_diagonal.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/virtual_partitioning.hpp"
#include "dtdma/virtual_reduced_system.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

namespace detail {

inline TridiagonalShared make_local_operator(
    const TridiagonalShared& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  TridiagonalShared local(local_row_count);
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    local.lower(local_row) = original.lower(global_row);
    local.diagonal(local_row) = original.diagonal(global_row);
    local.upper(local_row) = original.upper(global_row);
  }
  return local;
}

inline TridiagonalShiftedDiagonal make_local_operator(
    const TridiagonalShiftedDiagonal& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  TridiagonalShiftedDiagonal local(local_row_count,
                                   original.system_count());
  for (std::size_t system = 0; system < original.system_count(); ++system) {
    local.shift(system) = original.shift(system);
  }
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    local.lower(local_row) = original.lower(global_row);
    local.base_diagonal(local_row) = original.base_diagonal(global_row);
    local.upper(local_row) = original.upper(global_row);
  }
  return local;
}

inline TridiagonalSystemDiagonal make_local_operator(
    const TridiagonalSystemDiagonal& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  TridiagonalSystemDiagonal local(local_row_count,
                                  original.system_count());
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    local.lower(local_row) = original.lower(global_row);
    local.upper(local_row) = original.upper(global_row);
    for (std::size_t system = 0; system < original.system_count(); ++system) {
      local.diagonal(local_row, system) =
          original.diagonal(global_row, system);
    }
  }
  return local;
}

inline TridiagonalBatch make_local_operator(
    const TridiagonalBatch& original,
    const std::size_t begin_row,
    const std::size_t local_row_count) {
  TridiagonalBatch local(local_row_count, original.system_count());
  for (std::size_t local_row = 0; local_row < local_row_count; ++local_row) {
    const std::size_t global_row = begin_row + local_row;
    for (std::size_t system = 0; system < original.system_count(); ++system) {
      local.lower(local_row, system) = original.lower(global_row, system);
      local.diagonal(local_row, system) =
          original.diagonal(global_row, system);
      local.upper(local_row, system) = original.upper(global_row, system);
    }
  }
  return local;
}

inline SharedPreparedOperator make_prepared_operator(
    const TridiagonalShared& original) {
  return SharedPreparedOperator(original.row_count());
}

inline PreparedOperatorBatch make_prepared_operator(
    const TridiagonalShiftedDiagonal& original) {
  return PreparedOperatorBatch(original.row_count(), original.system_count());
}

inline PreparedOperatorBatch make_prepared_operator(
    const TridiagonalSystemDiagonal& original) {
  return PreparedOperatorBatch(original.row_count(), original.system_count());
}

inline PreparedOperatorBatch make_prepared_operator(
    const TridiagonalBatch& original) {
  return PreparedOperatorBatch(original.row_count(), original.system_count());
}

inline TridiagonalShared make_reduced_operator(
    const TridiagonalShared&,
    const std::size_t row_count) {
  return TridiagonalShared(row_count);
}

inline TridiagonalBatch make_reduced_operator(
    const TridiagonalShiftedDiagonal& original,
    const std::size_t row_count) {
  return TridiagonalBatch(row_count, original.system_count());
}

inline TridiagonalBatch make_reduced_operator(
    const TridiagonalSystemDiagonal& original,
    const std::size_t row_count) {
  return TridiagonalBatch(row_count, original.system_count());
}

inline TridiagonalBatch make_reduced_operator(
    const TridiagonalBatch& original,
    const std::size_t row_count) {
  return TridiagonalBatch(row_count, original.system_count());
}

}  // namespace detail

template <typename OriginalOperator>
inline void solve_virtual_system(
    const OriginalOperator& original,
    const RhsBatch& input_rhs,
    const VirtualPartitioning& partitioning,
    RhsBatch& solution) {
  if (partitioning.global_row_count() != original.row_count() ||
      input_rhs.row_count() != original.row_count() ||
      solution.row_count() != original.row_count() ||
      solution.batch_size() != input_rhs.batch_size() ||
      !detail::rhs_batch_size_is_compatible(original,
                                             input_rhs.batch_size())) {
    throw std::invalid_argument(
        "global operator, partitioning, and solution dimensions must match");
  }

  const std::size_t partition_count = partitioning.partition_count();
  const std::size_t batch_size = input_rhs.batch_size();
  using PreparedOperator = typename OriginalOperator::PreparedOperator;
  using ReducedOperator = typename OriginalOperator::ReducedOperator;

  std::vector<PreparedOperator> prepared_partitions;
  std::vector<RhsBatch> working_partitions;
  std::vector<EndpointBatch> partition_endpoints;
  prepared_partitions.reserve(partition_count);
  working_partitions.reserve(partition_count);
  partition_endpoints.reserve(partition_count);

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    const auto partition = partitioning.partition(rank);
    const std::size_t local_row_count = partition.local_row_count;
    auto local_original = detail::make_local_operator(
        original, partition.begin_row, local_row_count);

    prepared_partitions.push_back(
        detail::make_prepared_operator(local_original));
    working_partitions.emplace_back(local_row_count, batch_size);
    partition_endpoints.emplace_back(batch_size);

    RhsBatch local_input(local_row_count, batch_size);
    for (std::size_t local_row = 0; local_row < local_row_count;
         ++local_row) {
      const std::size_t global_row = partition.begin_row + local_row;
      for (std::size_t system = 0; system < batch_size; ++system) {
        local_input.rhs(local_row, system) =
            input_rhs.rhs(global_row, system);
      }
    }

    prepare_forward_coefficients(local_original,
                                 prepared_partitions.back());
    prepare_backward_coefficients(local_original,
                                  prepared_partitions.back());
    initialize_reduced_rhs(local_input, working_partitions.back());
    reduce_rhs_forward(local_original, prepared_partitions.back(),
                       working_partitions.back());
    reduce_rhs_backward(local_original, prepared_partitions.back(),
                        working_partitions.back());
    extract_reduced_rhs_endpoints(working_partitions.back(),
                                  partition_endpoints.back());
  }

  ReducedOperator reduced_system =
      detail::make_reduced_operator(original, 2 * partition_count);
  RhsBatch reduced_rhs(2 * partition_count, batch_size);
  assemble_virtual_reduced_system(
      partitioning,
      std::span<const PreparedOperator>(prepared_partitions),
      partition_endpoints, reduced_system, reduced_rhs);
  batched_thomas_solve(reduced_system, reduced_rhs);
  recover_virtual_reduced_endpoints(partitioning, reduced_system, reduced_rhs,
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
