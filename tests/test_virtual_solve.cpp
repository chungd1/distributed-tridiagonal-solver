#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/backward_rhs_reduction.hpp"
#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/single_partition_reduced_system.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/virtual_partitioning.hpp"
#include "dtdma/virtual_reduced_system.hpp"
#include "dtdma/virtual_solve.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

dtdma::TridiagonalBatch assemble_reduced_system_for_test(
    const dtdma::TridiagonalBatch& original,
    const dtdma::VirtualPartitioning& partitioning) {
  const std::size_t partition_count = partitioning.partition_count();
  const std::size_t batch_size = original.batch_size();
  std::vector<dtdma::PreparedOperatorBatch> prepared;
  std::vector<dtdma::ReducedRhsBatch> working;
  std::vector<dtdma::ReducedRhsEndpoints> endpoints;
  prepared.reserve(partition_count);
  working.reserve(partition_count);
  endpoints.reserve(partition_count);

  for (std::size_t rank = 0; rank < partition_count; ++rank) {
    const auto partition = partitioning.partition(rank);
    const std::size_t local_row_count = partition.local_row_count;
    dtdma::TridiagonalBatch local_original(local_row_count, batch_size);
    for (std::size_t local_row = 0; local_row < local_row_count;
         ++local_row) {
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

    prepared.emplace_back(local_row_count, batch_size);
    working.emplace_back(local_row_count, batch_size);
    endpoints.emplace_back(batch_size);
    dtdma::prepare_forward_coefficients(local_original, prepared.back());
    dtdma::prepare_backward_coefficients(local_original, prepared.back());
    dtdma::initialize_reduced_rhs(local_original, working.back());
    dtdma::reduce_rhs_forward(local_original, prepared.back(),
                              working.back());
    dtdma::reduce_rhs_backward(local_original, prepared.back(),
                               working.back());
    dtdma::extract_reduced_rhs_endpoints(working.back(), endpoints.back());
  }

  dtdma::TridiagonalBatch reduced_system(2 * partition_count, batch_size);
  dtdma::assemble_virtual_reduced_system(
      partitioning, prepared, endpoints, reduced_system);
  return reduced_system;
}

}  // namespace

TEST_CASE("Balanced virtual partitioning describes contiguous uneven slices") {
  const dtdma::VirtualPartitioning partitioning(17, 4);
  const std::array<std::size_t, 4> expected_begin{0, 5, 9, 13};
  const std::array<std::size_t, 4> expected_end{5, 9, 13, 17};
  const std::array<std::size_t, 4> expected_count{5, 4, 4, 4};

  CHECK(partitioning.global_row_count() == 17);
  CHECK(partitioning.partition_count() == 4);
  std::size_t next_row = 0;
  for (std::size_t rank = 0; rank < partitioning.partition_count(); ++rank) {
    const auto partition = partitioning.partition(rank);
    CHECK(partition.rank == rank);
    CHECK(partition.begin_row == expected_begin[rank]);
    CHECK(partition.end_row == expected_end[rank]);
    CHECK(partition.local_row_count == expected_count[rank]);
    CHECK(partition.begin_row == next_row);
    next_row = partition.end_row;
  }
  CHECK(next_row == partitioning.global_row_count());
  CHECK_THROWS_AS(partitioning.partition(4), std::out_of_range);
}

TEST_CASE("Balanced virtual partitioning handles remainder and minimum sizes") {
  const dtdma::VirtualPartitioning fourteen_rows(14, 3);
  CHECK(fourteen_rows.partition(0).local_row_count == 5);
  CHECK(fourteen_rows.partition(1).local_row_count == 5);
  CHECK(fourteen_rows.partition(2).local_row_count == 4);

  const dtdma::VirtualPartitioning thirteen_rows(13, 4);
  CHECK(thirteen_rows.partition(0).local_row_count == 4);
  CHECK(thirteen_rows.partition(1).local_row_count == 3);
  CHECK(thirteen_rows.partition(2).local_row_count == 3);
  CHECK(thirteen_rows.partition(3).local_row_count == 3);

  const dtdma::VirtualPartitioning minimum_rows(12, 4);
  for (std::size_t rank = 0; rank < 4; ++rank) {
    CHECK(minimum_rows.partition(rank).local_row_count == 3);
  }
}

TEST_CASE("Virtual partitioning rejects unsupported decompositions") {
  CHECK_THROWS_AS(dtdma::VirtualPartitioning(24, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::VirtualPartitioning(24, 5),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::VirtualPartitioning(11, 4),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::VirtualPartitioning(0, 1),
                  std::invalid_argument);
}

TEST_CASE("Virtual reduced assembly follows rank endpoint ordering") {
  const dtdma::VirtualPartitioning partitioning(6, 2);
  std::vector<dtdma::PreparedOperatorBatch> prepared;
  std::vector<dtdma::ReducedRhsEndpoints> endpoints;
  prepared.emplace_back(3, 1);
  prepared.emplace_back(3, 1);
  endpoints.emplace_back(1);
  endpoints.emplace_back(1);

  prepared[0].prepared_lower(0, 0) = 99.0F;
  prepared[0].prepared_diagonal(0, 0) = 10.0F;
  prepared[0].prepared_upper(0, 0) = 3.0F;
  prepared[0].prepared_lower(2, 0) = 4.0F;
  prepared[0].prepared_diagonal(2, 0) = 20.0F;
  prepared[0].prepared_upper(2, 0) = 5.0F;
  prepared[1].prepared_lower(0, 0) = 6.0F;
  prepared[1].prepared_diagonal(0, 0) = 30.0F;
  prepared[1].prepared_upper(0, 0) = 7.0F;
  prepared[1].prepared_lower(2, 0) = 8.0F;
  prepared[1].prepared_diagonal(2, 0) = 40.0F;
  prepared[1].prepared_upper(2, 0) = 99.0F;
  endpoints[0].endpoint(0, 0) = 1.0F;
  endpoints[0].endpoint(0, 1) = 2.0F;
  endpoints[1].endpoint(0, 0) = 3.0F;
  endpoints[1].endpoint(0, 1) = 4.0F;

  std::array<std::vector<dtdma::Scalar>, 2> preserved_prepared_lower;
  std::array<std::vector<dtdma::Scalar>, 2> preserved_prepared_diagonal;
  std::array<std::vector<dtdma::Scalar>, 2> preserved_prepared_upper;
  for (std::size_t rank = 0; rank < 2; ++rank) {
    preserved_prepared_lower[rank] = std::vector<dtdma::Scalar>(
        prepared[rank].prepared_lower().begin(),
        prepared[rank].prepared_lower().end());
    preserved_prepared_diagonal[rank] = std::vector<dtdma::Scalar>(
        prepared[rank].prepared_diagonal().begin(),
        prepared[rank].prepared_diagonal().end());
    preserved_prepared_upper[rank] = std::vector<dtdma::Scalar>(
        prepared[rank].prepared_upper().begin(),
        prepared[rank].prepared_upper().end());
  }
  const std::vector<dtdma::Scalar> endpoint_zero(
      endpoints[0].endpoints().begin(), endpoints[0].endpoints().end());
  const std::vector<dtdma::Scalar> endpoint_one(
      endpoints[1].endpoints().begin(), endpoints[1].endpoints().end());
  dtdma::TridiagonalBatch reduced_system(4, 1);

  dtdma::assemble_virtual_reduced_system(
      partitioning, prepared, endpoints, reduced_system);

  const std::array<dtdma::Scalar, 4> expected_lower{0.0F, 4.0F, 6.0F,
                                                    8.0F};
  const std::array<dtdma::Scalar, 4> expected_diagonal{10.0F, 20.0F,
                                                       30.0F, 40.0F};
  const std::array<dtdma::Scalar, 4> expected_upper{3.0F, 5.0F, 7.0F,
                                                    0.0F};
  const std::array<dtdma::Scalar, 4> expected_rhs{1.0F, 2.0F, 3.0F,
                                                  4.0F};
  for (std::size_t row = 0; row < reduced_system.row_count(); ++row) {
    CHECK(reduced_system.lower(row, 0) == expected_lower[row]);
    CHECK(reduced_system.diagonal(row, 0) == expected_diagonal[row]);
    CHECK(reduced_system.upper(row, 0) == expected_upper[row]);
    CHECK(reduced_system.rhs(row, 0) == expected_rhs[row]);
  }
  for (std::size_t rank = 0; rank < 2; ++rank) {
    CHECK(std::vector<dtdma::Scalar>(
              prepared[rank].prepared_lower().begin(),
              prepared[rank].prepared_lower().end()) ==
          preserved_prepared_lower[rank]);
    CHECK(std::vector<dtdma::Scalar>(
              prepared[rank].prepared_diagonal().begin(),
              prepared[rank].prepared_diagonal().end()) ==
          preserved_prepared_diagonal[rank]);
    CHECK(std::vector<dtdma::Scalar>(
              prepared[rank].prepared_upper().begin(),
              prepared[rank].prepared_upper().end()) ==
          preserved_prepared_upper[rank]);
  }
  CHECK(std::vector<dtdma::Scalar>(endpoints[0].endpoints().begin(),
                                   endpoints[0].endpoints().end()) ==
        endpoint_zero);
  CHECK(std::vector<dtdma::Scalar>(endpoints[1].endpoints().begin(),
                                   endpoints[1].endpoints().end()) ==
        endpoint_one);
}

TEST_CASE("One virtual partition matches single-partition assembly") {
  const dtdma::VirtualPartitioning partitioning(5, 1);
  dtdma::TridiagonalBatch original(5, 1);
  std::vector<dtdma::PreparedOperatorBatch> prepared;
  std::vector<dtdma::ReducedRhsBatch> working;
  std::vector<dtdma::ReducedRhsEndpoints> endpoints;
  prepared.emplace_back(5, 1);
  working.emplace_back(5, 1);
  endpoints.emplace_back(1);

  prepared[0].prepared_lower(0, 0) = 2.0F;
  prepared[0].prepared_diagonal(0, 0) = 10.0F;
  prepared[0].prepared_upper(0, 0) = 3.0F;
  prepared[0].prepared_lower(4, 0) = 4.0F;
  prepared[0].prepared_diagonal(4, 0) = 20.0F;
  prepared[0].prepared_upper(4, 0) = 5.0F;
  endpoints[0].endpoint(0, 0) = 7.0F;
  endpoints[0].endpoint(0, 1) = 9.0F;
  dtdma::TridiagonalBatch single_reduced(2, 1);
  dtdma::TridiagonalBatch virtual_reduced(2, 1);

  dtdma::assemble_single_partition_reduced_system(
      original, prepared[0], working[0], endpoints[0], single_reduced);
  dtdma::assemble_virtual_reduced_system(
      partitioning, prepared, endpoints, virtual_reduced);

  for (std::size_t row = 0; row < 2; ++row) {
    CHECK(virtual_reduced.lower(row, 0) == single_reduced.lower(row, 0));
    CHECK(virtual_reduced.diagonal(row, 0) ==
          single_reduced.diagonal(row, 0));
    CHECK(virtual_reduced.upper(row, 0) == single_reduced.upper(row, 0));
    CHECK(virtual_reduced.rhs(row, 0) == single_reduced.rhs(row, 0));
  }
  CHECK(virtual_reduced.lower(0, 0) == 0.0F);
  CHECK(virtual_reduced.diagonal(0, 0) == 10.0F);
  CHECK(virtual_reduced.upper(1, 0) == 0.0F);
  CHECK(virtual_reduced.diagonal(1, 0) == 20.0F);
}

TEST_CASE("Physical exterior sentinels are ignored for every partition count") {
  constexpr std::size_t row_count = 12;
  constexpr dtdma::Scalar tolerance = 2.0e-5F;
  dtdma::TridiagonalBatch valid(row_count, 1);
  for (std::size_t row = 0; row < row_count; ++row) {
    const dtdma::Scalar exact = static_cast<dtdma::Scalar>(row + 1);
    valid.lower(row, 0) = row == 0 ? 0.0F : -1.0F;
    valid.diagonal(row, 0) = 4.0F;
    valid.upper(row, 0) =
        row + 1 == row_count ? 0.0F : -1.0F;
    valid.rhs(row, 0) = 4.0F * exact;
    if (row > 0) {
      valid.rhs(row, 0) -= static_cast<dtdma::Scalar>(row);
    }
    if (row + 1 < row_count) {
      valid.rhs(row, 0) -= static_cast<dtdma::Scalar>(row + 2);
    }
  }

  dtdma::TridiagonalBatch sentinel = valid;
  sentinel.lower(0, 0) = 99.0F;
  sentinel.upper(row_count - 1, 0) = -99.0F;
  const std::vector<dtdma::Scalar> preserved_lower(
      sentinel.lower().begin(), sentinel.lower().end());
  const std::vector<dtdma::Scalar> preserved_diagonal(
      sentinel.diagonal().begin(), sentinel.diagonal().end());
  const std::vector<dtdma::Scalar> preserved_upper(
      sentinel.upper().begin(), sentinel.upper().end());
  const std::vector<dtdma::Scalar> preserved_rhs(
      sentinel.rhs().begin(), sentinel.rhs().end());

  dtdma::TridiagonalBatch global_reference = valid;
  dtdma::batched_thomas_solve(global_reference);
  for (std::size_t partition_count = 1; partition_count <= 4;
       ++partition_count) {
    const dtdma::VirtualPartitioning partitioning(
        row_count, partition_count);
    const dtdma::TridiagonalBatch valid_reduced =
        assemble_reduced_system_for_test(valid, partitioning);
    const dtdma::TridiagonalBatch sentinel_reduced =
        assemble_reduced_system_for_test(sentinel, partitioning);

    CHECK(std::vector<dtdma::Scalar>(sentinel_reduced.lower().begin(),
                                     sentinel_reduced.lower().end()) ==
          std::vector<dtdma::Scalar>(valid_reduced.lower().begin(),
                                     valid_reduced.lower().end()));
    CHECK(std::vector<dtdma::Scalar>(sentinel_reduced.diagonal().begin(),
                                     sentinel_reduced.diagonal().end()) ==
          std::vector<dtdma::Scalar>(valid_reduced.diagonal().begin(),
                                     valid_reduced.diagonal().end()));
    CHECK(std::vector<dtdma::Scalar>(sentinel_reduced.upper().begin(),
                                     sentinel_reduced.upper().end()) ==
          std::vector<dtdma::Scalar>(valid_reduced.upper().begin(),
                                     valid_reduced.upper().end()));
    CHECK(std::vector<dtdma::Scalar>(sentinel_reduced.rhs().begin(),
                                     sentinel_reduced.rhs().end()) ==
          std::vector<dtdma::Scalar>(valid_reduced.rhs().begin(),
                                     valid_reduced.rhs().end()));
    CHECK(sentinel_reduced.lower(0, 0) == 0.0F);
    CHECK(sentinel_reduced.upper(2 * partition_count - 1, 0) == 0.0F);
    for (std::size_t rank = 0; rank + 1 < partition_count; ++rank) {
      CHECK(sentinel_reduced.upper(2 * rank + 1, 0) == -1.0F);
      CHECK(sentinel_reduced.lower(2 * rank + 2, 0) == -1.0F);
    }

    dtdma::ReducedRhsBatch valid_solution(row_count, 1);
    dtdma::ReducedRhsBatch sentinel_solution(row_count, 1);
    dtdma::solve_virtual_system(valid, partitioning, valid_solution);
    dtdma::solve_virtual_system(sentinel, partitioning,
                                sentinel_solution);
    for (std::size_t row = 0; row < row_count; ++row) {
      CHECK(sentinel_solution.rhs(row, 0) == valid_solution.rhs(row, 0));
      CHECK(sentinel_solution.rhs(row, 0) ==
            Catch::Approx(global_reference.rhs(row, 0)).margin(tolerance));
    }
  }

  CHECK(std::vector<dtdma::Scalar>(sentinel.lower().begin(),
                                   sentinel.lower().end()) ==
        preserved_lower);
  CHECK(std::vector<dtdma::Scalar>(sentinel.diagonal().begin(),
                                   sentinel.diagonal().end()) ==
        preserved_diagonal);
  CHECK(std::vector<dtdma::Scalar>(sentinel.upper().begin(),
                                   sentinel.upper().end()) ==
        preserved_upper);
  CHECK(std::vector<dtdma::Scalar>(sentinel.rhs().begin(),
                                   sentinel.rhs().end()) == preserved_rhs);
}

TEST_CASE("Virtual solves match global Thomas for every partition count") {
  constexpr std::size_t row_count = 24;
  constexpr dtdma::Scalar tolerance = 4.0e-5F;
  dtdma::TridiagonalBatch original(row_count, 1);
  std::array<dtdma::Scalar, row_count> exact_solution{};

  for (std::size_t row = 0; row < row_count; ++row) {
    exact_solution[row] = static_cast<dtdma::Scalar>(row + 1);
    original.lower(row, 0) =
        row == 0 ? 0.0F : -0.35F - 0.01F * static_cast<float>(row % 5);
    original.diagonal(row, 0) =
        4.5F + 0.1F * static_cast<float>(row % 7);
    original.upper(row, 0) =
        row + 1 == row_count
            ? 0.0F
            : 0.25F + 0.02F * static_cast<float>(row % 3);
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    dtdma::Scalar rhs = original.diagonal(row, 0) * exact_solution[row];
    if (row > 0) {
      rhs += original.lower(row, 0) * exact_solution[row - 1];
    }
    if (row + 1 < row_count) {
      rhs += original.upper(row, 0) * exact_solution[row + 1];
    }
    original.rhs(row, 0) = rhs;
  }

  const std::vector<dtdma::Scalar> preserved_lower(original.lower().begin(),
                                                    original.lower().end());
  const std::vector<dtdma::Scalar> preserved_diagonal(
      original.diagonal().begin(), original.diagonal().end());
  const std::vector<dtdma::Scalar> preserved_upper(original.upper().begin(),
                                                    original.upper().end());
  const std::vector<dtdma::Scalar> preserved_rhs(original.rhs().begin(),
                                                  original.rhs().end());
  dtdma::TridiagonalBatch global_reference = original;
  dtdma::batched_thomas_solve(global_reference);
  std::array<std::vector<dtdma::Scalar>, 4> partition_solutions;

  for (std::size_t partition_count = 1;
       partition_count <= 4; ++partition_count) {
    const dtdma::VirtualPartitioning partitioning(
        row_count, partition_count);
    dtdma::ReducedRhsBatch solution(row_count, 1);
    dtdma::solve_virtual_system(original, partitioning, solution);
    partition_solutions[partition_count - 1] =
        std::vector<dtdma::Scalar>(solution.rhs().begin(),
                                   solution.rhs().end());

    for (std::size_t rank = 0; rank < partition_count; ++rank) {
      const auto partition = partitioning.partition(rank);
      CHECK(solution.rhs(partition.begin_row, 0) ==
            Catch::Approx(exact_solution[partition.begin_row])
                .margin(tolerance));
      CHECK(solution.rhs(partition.end_row - 1, 0) ==
            Catch::Approx(exact_solution[partition.end_row - 1])
                .margin(tolerance));
    }
    for (std::size_t row = 0; row < row_count; ++row) {
      CHECK(solution.rhs(row, 0) ==
            Catch::Approx(exact_solution[row]).margin(tolerance));
      CHECK(solution.rhs(row, 0) ==
            Catch::Approx(global_reference.rhs(row, 0)).margin(tolerance));
    }
  }

  for (std::size_t partition_count = 1;
       partition_count < 4; ++partition_count) {
    for (std::size_t row = 0; row < row_count; ++row) {
      CHECK(partition_solutions[partition_count][row] ==
            Catch::Approx(partition_solutions[0][row]).margin(tolerance));
    }
  }

  CHECK(std::vector<dtdma::Scalar>(original.lower().begin(),
                                   original.lower().end()) == preserved_lower);
  CHECK(std::vector<dtdma::Scalar>(original.diagonal().begin(),
                                   original.diagonal().end()) ==
        preserved_diagonal);
  CHECK(std::vector<dtdma::Scalar>(original.upper().begin(),
                                   original.upper().end()) == preserved_upper);
  CHECK(std::vector<dtdma::Scalar>(original.rhs().begin(),
                                   original.rhs().end()) == preserved_rhs);
}

TEST_CASE("Batched virtual solves keep systems independent for every partition count") {
  constexpr std::size_t row_count = 17;
  constexpr std::size_t batch_size = 3;
  constexpr dtdma::Scalar tolerance = 4.0e-5F;
  dtdma::TridiagonalBatch original(row_count, batch_size);
  std::array<dtdma::Scalar, row_count * batch_size> exact_solution{};

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const std::size_t index = row * batch_size + system;
      if (system == 0) {
        exact_solution[index] = static_cast<dtdma::Scalar>(row + 1);
      } else if (system == 1) {
        exact_solution[index] =
            -0.5F * static_cast<dtdma::Scalar>(row + 1);
      } else {
        exact_solution[index] =
            2.0F + 0.25F * static_cast<dtdma::Scalar>(row);
      }
      original.lower(row, system) =
          row == 0
              ? 0.0F
              : -0.2F - 0.03F * static_cast<float>((row + system) % 4);
      original.diagonal(row, system) =
          4.0F + 0.2F * static_cast<float>((row + 2 * system) % 6);
      original.upper(row, system) =
          row + 1 == row_count
              ? 0.0F
              : 0.15F +
                    0.02F * static_cast<float>((2 * row + system) % 5);
    }
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const std::size_t index = row * batch_size + system;
      dtdma::Scalar rhs =
          original.diagonal(row, system) * exact_solution[index];
      if (row > 0) {
        rhs += original.lower(row, system) *
               exact_solution[(row - 1) * batch_size + system];
      }
      if (row + 1 < row_count) {
        rhs += original.upper(row, system) *
               exact_solution[(row + 1) * batch_size + system];
      }
      original.rhs(row, system) = rhs;
    }
  }

  dtdma::TridiagonalBatch global_reference = original;
  dtdma::batched_thomas_solve(global_reference);
  std::array<std::vector<dtdma::Scalar>, 4> partition_solutions;
  for (std::size_t partition_count = 1;
       partition_count <= 4; ++partition_count) {
    const dtdma::VirtualPartitioning partitioning(
        row_count, partition_count);
    dtdma::ReducedRhsBatch solution(row_count, batch_size);
    dtdma::solve_virtual_system(original, partitioning, solution);
    partition_solutions[partition_count - 1] =
        std::vector<dtdma::Scalar>(solution.rhs().begin(),
                                   solution.rhs().end());

    for (std::size_t row = 0; row < row_count; ++row) {
      for (std::size_t system = 0; system < batch_size; ++system) {
        const std::size_t index = row * batch_size + system;
        CHECK(solution.rhs(row, system) ==
              Catch::Approx(exact_solution[index]).margin(tolerance));
        CHECK(solution.rhs(row, system) ==
              Catch::Approx(global_reference.rhs(row, system))
                  .margin(tolerance));
      }
    }
    for (std::size_t rank = 0; rank < partition_count; ++rank) {
      const auto partition = partitioning.partition(rank);
      for (std::size_t system = 0; system < batch_size; ++system) {
        const std::size_t first_index =
            partition.begin_row * batch_size + system;
        const std::size_t last_index =
            (partition.end_row - 1) * batch_size + system;
        CHECK(solution.rhs(partition.begin_row, system) ==
              Catch::Approx(global_reference.rhs(partition.begin_row,
                                                  system))
                  .margin(tolerance));
        CHECK(solution.rhs(partition.end_row - 1, system) ==
              Catch::Approx(global_reference.rhs(partition.end_row - 1,
                                                  system))
                  .margin(tolerance));
        CHECK(solution.rhs(partition.begin_row, system) ==
              Catch::Approx(exact_solution[first_index]).margin(tolerance));
        CHECK(solution.rhs(partition.end_row - 1, system) ==
              Catch::Approx(exact_solution[last_index]).margin(tolerance));
      }
    }
  }

  for (std::size_t partition_count = 1;
       partition_count < 4; ++partition_count) {
    for (std::size_t index = 0;
         index < row_count * batch_size; ++index) {
      CHECK(partition_solutions[partition_count][index] ==
            Catch::Approx(partition_solutions[0][index]).margin(tolerance));
    }
  }
}

TEST_CASE("Virtual reduced operations reject incompatible dimensions") {
  const dtdma::VirtualPartitioning partitioning(7, 2);
  std::vector<dtdma::PreparedOperatorBatch> prepared;
  std::vector<dtdma::ReducedRhsEndpoints> endpoints;
  prepared.emplace_back(4, 1);
  prepared.emplace_back(3, 1);
  endpoints.emplace_back(1);
  endpoints.emplace_back(1);
  dtdma::TridiagonalBatch reduced(4, 1);
  dtdma::TridiagonalBatch wrong_reduced_rows(3, 1);
  dtdma::TridiagonalBatch wrong_reduced_batch(4, 2);

  CHECK_THROWS_AS(dtdma::assemble_virtual_reduced_system(
                      partitioning,
                      std::span<const dtdma::PreparedOperatorBatch>(
                          prepared.data(), 1),
                      endpoints, reduced),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_virtual_reduced_system(
                      partitioning, prepared, endpoints,
                      wrong_reduced_rows),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_virtual_reduced_system(
                      partitioning, prepared, endpoints,
                      wrong_reduced_batch),
                  std::invalid_argument);

  std::vector<dtdma::ReducedRhsEndpoints> wrong_endpoint_batch;
  wrong_endpoint_batch.emplace_back(1);
  wrong_endpoint_batch.emplace_back(2);
  CHECK_THROWS_AS(dtdma::recover_virtual_reduced_endpoints(
                      partitioning, reduced, wrong_endpoint_batch),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::recover_virtual_reduced_endpoints(
                      partitioning, wrong_reduced_rows, endpoints),
                  std::invalid_argument);
}

TEST_CASE("Virtual solve rejects incompatible global dimensions") {
  const dtdma::TridiagonalBatch original(24, 2);
  const dtdma::VirtualPartitioning wrong_global_rows(12, 2);
  const dtdma::VirtualPartitioning partitioning(24, 2);
  dtdma::ReducedRhsBatch solution(24, 2);
  dtdma::ReducedRhsBatch wrong_solution_rows(12, 2);
  dtdma::ReducedRhsBatch wrong_solution_batch(24, 1);

  CHECK_THROWS_AS(dtdma::solve_virtual_system(
                      original, wrong_global_rows, solution),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::solve_virtual_system(
                      original, partitioning, wrong_solution_rows),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::solve_virtual_system(
                      original, partitioning, wrong_solution_batch),
                  std::invalid_argument);
}
