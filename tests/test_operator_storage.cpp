#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/backward_rhs_reduction.hpp"
#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/reduced_rhs_endpoints.hpp"
#include "dtdma/shared_prepared_operator.hpp"
#include "dtdma/shared_tridiagonal_batch.hpp"
#include "dtdma/shifted_diagonal_tridiagonal_batch.hpp"
#include "dtdma/single_partition_reconstruction.hpp"
#include "dtdma/single_partition_reduced_system.hpp"
#include "dtdma/system_diagonal_tridiagonal_batch.hpp"
#include "dtdma/virtual_partitioning.hpp"
#include "dtdma/virtual_solve.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace {

constexpr dtdma::Scalar tolerance = 2.0e-5F;

struct PipelineSnapshot {
  std::vector<dtdma::Scalar> prepared_lower;
  std::vector<dtdma::Scalar> prepared_diagonal;
  std::vector<dtdma::Scalar> prepared_upper;
  std::vector<dtdma::Scalar> reduced_rhs;
  std::vector<dtdma::Scalar> reduced_lower;
  std::vector<dtdma::Scalar> reduced_diagonal;
  std::vector<dtdma::Scalar> reduced_upper;
  std::vector<dtdma::Scalar> reduced_system_rhs;
  std::vector<dtdma::Scalar> solved_endpoints;
  std::vector<dtdma::Scalar> solution;
  std::size_t prepared_physical_element_count{};
  std::size_t reduced_coefficient_element_count{};
  std::size_t reduced_rhs_element_count{};
};

template <typename OriginalOperator>
void construct_rhs(OriginalOperator& original,
                   const std::span<const dtdma::Scalar> exact) {
  const OriginalOperator& coefficients = original;
  for (std::size_t row = 0; row < original.row_count(); ++row) {
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      const std::size_t index = row * original.batch_size() + system;
      dtdma::Scalar value =
          coefficients.diagonal(row, system) * exact[index];
      if (row > 0) {
        value += coefficients.lower(row, system) *
                 exact[(row - 1) * original.batch_size() + system];
      }
      if (row + 1 < original.row_count()) {
        value += coefficients.upper(row, system) *
                 exact[(row + 1) * original.batch_size() + system];
      }
      original.rhs(row, system) = value;
    }
  }
}

template <typename OriginalOperator>
dtdma::TridiagonalBatch materialize_fully_batched(
    const OriginalOperator& original) {
  dtdma::TridiagonalBatch result(original.row_count(),
                                  original.batch_size());
  for (std::size_t row = 0; row < original.row_count(); ++row) {
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      result.lower(row, system) = original.lower(row, system);
      result.diagonal(row, system) = original.diagonal(row, system);
      result.upper(row, system) = original.upper(row, system);
      result.rhs(row, system) = original.rhs(row, system);
    }
  }
  return result;
}

template <typename OriginalOperator>
PipelineSnapshot run_single_partition_pipeline(
    const OriginalOperator& original) {
  using PreparedOperator = typename OriginalOperator::PreparedOperator;
  using ReducedOperator = typename OriginalOperator::ReducedOperator;

  const std::size_t row_count = original.row_count();
  const std::size_t batch_size = original.batch_size();
  PreparedOperator prepared(row_count, batch_size);
  dtdma::prepare_forward_coefficients(original, prepared);
  dtdma::prepare_backward_coefficients(original, prepared);

  PipelineSnapshot snapshot;
  snapshot.prepared_physical_element_count = prepared.element_count();
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      snapshot.prepared_lower.push_back(
          prepared.prepared_lower(row, system));
      snapshot.prepared_diagonal.push_back(
          prepared.prepared_diagonal(row, system));
      snapshot.prepared_upper.push_back(
          prepared.prepared_upper(row, system));
    }
  }

  dtdma::ReducedRhsBatch working(row_count, batch_size);
  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);
  dtdma::reduce_rhs_backward(original, prepared, working);
  snapshot.reduced_rhs.assign(working.rhs().begin(), working.rhs().end());

  dtdma::ReducedRhsEndpoints endpoints(batch_size);
  dtdma::extract_reduced_rhs_endpoints(working, endpoints);
  ReducedOperator reduced(2, batch_size);
  dtdma::assemble_single_partition_reduced_system(
      original, prepared, working, endpoints, reduced);
  snapshot.reduced_coefficient_element_count = reduced.lower().size();
  snapshot.reduced_rhs_element_count = reduced.rhs().size();
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      snapshot.reduced_lower.push_back(reduced.lower(row, system));
      snapshot.reduced_diagonal.push_back(reduced.diagonal(row, system));
      snapshot.reduced_upper.push_back(reduced.upper(row, system));
      snapshot.reduced_system_rhs.push_back(reduced.rhs(row, system));
    }
  }

  dtdma::batched_thomas_solve(reduced);
  dtdma::recover_single_partition_endpoints(reduced, endpoints);
  snapshot.solved_endpoints.assign(endpoints.endpoints().begin(),
                                   endpoints.endpoints().end());
  dtdma::reconstruct_single_partition(prepared, working, endpoints);
  snapshot.solution.assign(working.rhs().begin(), working.rhs().end());
  return snapshot;
}

void check_values(const std::vector<dtdma::Scalar>& actual,
                  const std::vector<dtdma::Scalar>& expected) {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    CHECK(actual[index] == Catch::Approx(expected[index]).margin(tolerance));
  }
}

void check_pipeline(const PipelineSnapshot& actual,
                    const PipelineSnapshot& expected) {
  check_values(actual.prepared_lower, expected.prepared_lower);
  check_values(actual.prepared_diagonal, expected.prepared_diagonal);
  check_values(actual.prepared_upper, expected.prepared_upper);
  check_values(actual.reduced_rhs, expected.reduced_rhs);
  check_values(actual.reduced_lower, expected.reduced_lower);
  check_values(actual.reduced_diagonal, expected.reduced_diagonal);
  check_values(actual.reduced_upper, expected.reduced_upper);
  check_values(actual.reduced_system_rhs, expected.reduced_system_rhs);
  check_values(actual.solved_endpoints, expected.solved_endpoints);
  check_values(actual.solution, expected.solution);
}

template <typename OriginalOperator>
std::vector<dtdma::Scalar> solve_with_prepared(
    const OriginalOperator& original,
    const typename OriginalOperator::PreparedOperator& prepared) {
  using ReducedOperator = typename OriginalOperator::ReducedOperator;
  dtdma::ReducedRhsBatch working(original.row_count(),
                                 original.batch_size());
  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);
  dtdma::reduce_rhs_backward(original, prepared, working);
  dtdma::ReducedRhsEndpoints endpoints(original.batch_size());
  dtdma::extract_reduced_rhs_endpoints(working, endpoints);
  ReducedOperator reduced(2, original.batch_size());
  dtdma::assemble_single_partition_reduced_system(
      original, prepared, working, endpoints, reduced);
  dtdma::batched_thomas_solve(reduced);
  dtdma::recover_single_partition_endpoints(reduced, endpoints);
  dtdma::reconstruct_single_partition(prepared, working, endpoints);
  return {working.rhs().begin(), working.rhs().end()};
}

template <typename OriginalOperator>
void check_virtual_solutions(const OriginalOperator& original,
                             const std::vector<dtdma::Scalar>& reference,
                             const std::span<const std::size_t> counts) {
  for (const std::size_t partition_count : counts) {
    const dtdma::VirtualPartitioning partitioning(original.row_count(),
                                                   partition_count);
    dtdma::ReducedRhsBatch solution(original.row_count(),
                                    original.batch_size());
    dtdma::solve_virtual_system(original, partitioning, solution);
    check_values({solution.rhs().begin(), solution.rhs().end()}, reference);
  }
}

std::vector<dtdma::Scalar> make_exact(const std::size_t row_count,
                                      const std::size_t batch_size) {
  std::vector<dtdma::Scalar> exact(row_count * batch_size);
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      exact[row * batch_size + system] =
          1.0F + static_cast<dtdma::Scalar>(row) +
          2.5F * static_cast<dtdma::Scalar>(system);
    }
  }
  return exact;
}

}  // namespace

TEST_CASE("Shared coefficient storage remains shared through every solve stage") {
  constexpr std::size_t row_count = 13;
  constexpr std::size_t batch_size = 3;
  const auto exact = make_exact(row_count, batch_size);

  dtdma::SharedTridiagonalBatch shared(row_count, batch_size);
  dtdma::ShiftedDiagonalTridiagonalBatch shifted(row_count, batch_size);
  dtdma::SystemDiagonalTridiagonalBatch system_diagonal(row_count,
                                                         batch_size);
  dtdma::TridiagonalBatch fully_batched(row_count, batch_size);
  for (std::size_t system = 0; system < batch_size; ++system) {
    shifted.shift(system) = 0.0F;
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    const dtdma::Scalar lower = row == 0 ? 0.0F : -0.45F - 0.01F * row;
    const dtdma::Scalar diagonal = 4.0F + 0.07F * row;
    const dtdma::Scalar upper =
        row + 1 == row_count ? 0.0F : -0.3F + 0.005F * row;
    shared.lower(row) = lower;
    shared.diagonal(row) = diagonal;
    shared.upper(row) = upper;
    shifted.lower(row) = lower;
    shifted.base_diagonal(row) = diagonal;
    shifted.upper(row) = upper;
    system_diagonal.lower(row) = lower;
    system_diagonal.upper(row) = upper;
    for (std::size_t system = 0; system < batch_size; ++system) {
      system_diagonal.diagonal(row, system) = diagonal;
      fully_batched.lower(row, system) = lower;
      fully_batched.diagonal(row, system) = diagonal;
      fully_batched.upper(row, system) = upper;
    }
  }
  construct_rhs(shared, exact);
  construct_rhs(shifted, exact);
  construct_rhs(system_diagonal, exact);
  construct_rhs(fully_batched, exact);

  const auto shared_result = run_single_partition_pipeline(shared);
  check_pipeline(run_single_partition_pipeline(shifted), shared_result);
  check_pipeline(run_single_partition_pipeline(system_diagonal),
                 shared_result);
  check_pipeline(run_single_partition_pipeline(fully_batched),
                 shared_result);

  CHECK(shared_result.prepared_physical_element_count == row_count);
  CHECK(shared_result.reduced_coefficient_element_count == 2);
  CHECK(shared_result.reduced_rhs_element_count == 2 * batch_size);
  CHECK(shared.lower().size() == row_count);
  CHECK(shared.diagonal().size() == row_count);
  CHECK(shared.upper().size() == row_count);
  CHECK(shared.rhs().size() == row_count * batch_size);
  check_values(shared_result.solution, exact);

  auto global_reference = materialize_fully_batched(shared);
  dtdma::batched_thomas_solve(global_reference);
  const std::vector<dtdma::Scalar> reference(global_reference.rhs().begin(),
                                              global_reference.rhs().end());
  check_values(shared_result.solution, reference);
  constexpr std::array<std::size_t, 4> partition_counts{1, 2, 3, 4};
  check_virtual_solutions(shared, reference, partition_counts);
  check_virtual_solutions(shifted, reference, partition_counts);
  check_virtual_solutions(system_diagonal, reference, partition_counts);
  check_virtual_solutions(fully_batched, reference, partition_counts);
}

TEST_CASE("Shifted diagonal storage agrees with materialized diagonal storage") {
  constexpr std::size_t row_count = 14;
  constexpr std::size_t batch_size = 3;
  const auto exact = make_exact(row_count, batch_size);
  dtdma::ShiftedDiagonalTridiagonalBatch shifted(row_count, batch_size);
  dtdma::SystemDiagonalTridiagonalBatch system_diagonal(row_count,
                                                         batch_size);
  dtdma::TridiagonalBatch fully_batched(row_count, batch_size);
  constexpr std::array<dtdma::Scalar, batch_size> shifts{-0.2F, 0.35F,
                                                         0.8F};
  for (std::size_t system = 0; system < batch_size; ++system) {
    shifted.shift(system) = shifts[system];
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    const dtdma::Scalar lower = row == 0 ? 0.0F : -0.4F - 0.008F * row;
    const dtdma::Scalar base_diagonal = 4.2F + 0.04F * row;
    const dtdma::Scalar upper =
        row + 1 == row_count ? 0.0F : -0.25F - 0.006F * row;
    shifted.lower(row) = lower;
    shifted.base_diagonal(row) = base_diagonal;
    shifted.upper(row) = upper;
    system_diagonal.lower(row) = lower;
    system_diagonal.upper(row) = upper;
    for (std::size_t system = 0; system < batch_size; ++system) {
      const dtdma::Scalar diagonal = base_diagonal + shifts[system];
      system_diagonal.diagonal(row, system) = diagonal;
      fully_batched.lower(row, system) = lower;
      fully_batched.diagonal(row, system) = diagonal;
      fully_batched.upper(row, system) = upper;
    }
  }
  construct_rhs(shifted, exact);
  construct_rhs(system_diagonal, exact);
  construct_rhs(fully_batched, exact);

  const auto shifted_result = run_single_partition_pipeline(shifted);
  check_pipeline(run_single_partition_pipeline(system_diagonal),
                 shifted_result);
  check_pipeline(run_single_partition_pipeline(fully_batched),
                 shifted_result);
  check_values(shifted_result.solution, exact);
  auto global_reference = materialize_fully_batched(shifted);
  dtdma::batched_thomas_solve(global_reference);
  const std::vector<dtdma::Scalar> reference(global_reference.rhs().begin(),
                                              global_reference.rhs().end());
  check_values(shifted_result.solution, reference);
  constexpr std::array<std::size_t, 2> partition_counts{3, 4};
  check_virtual_solutions(shifted, reference, partition_counts);
  check_virtual_solutions(system_diagonal, reference, partition_counts);
  check_virtual_solutions(fully_batched, reference, partition_counts);
}

TEST_CASE("Arbitrary system diagonals agree with fully batched storage") {
  constexpr std::size_t row_count = 17;
  constexpr std::size_t batch_size = 3;
  const auto exact = make_exact(row_count, batch_size);
  dtdma::SystemDiagonalTridiagonalBatch system_diagonal(row_count,
                                                         batch_size);
  dtdma::TridiagonalBatch fully_batched(row_count, batch_size);
  for (std::size_t row = 0; row < row_count; ++row) {
    const dtdma::Scalar lower = row == 0 ? 0.0F : -0.36F - 0.004F * row;
    const dtdma::Scalar upper =
        row + 1 == row_count ? 0.0F : -0.22F - 0.003F * row;
    system_diagonal.lower(row) = lower;
    system_diagonal.upper(row) = upper;
    for (std::size_t system = 0; system < batch_size; ++system) {
      const dtdma::Scalar diagonal =
          4.1F + 0.03F * row + 0.11F * system +
          0.007F * static_cast<dtdma::Scalar>(row * system);
      system_diagonal.diagonal(row, system) = diagonal;
      fully_batched.lower(row, system) = lower;
      fully_batched.diagonal(row, system) = diagonal;
      fully_batched.upper(row, system) = upper;
    }
  }
  construct_rhs(system_diagonal, exact);
  construct_rhs(fully_batched, exact);

  const auto system_result = run_single_partition_pipeline(system_diagonal);
  check_pipeline(run_single_partition_pipeline(fully_batched), system_result);
  check_values(system_result.solution, exact);
  auto global_reference = materialize_fully_batched(system_diagonal);
  dtdma::batched_thomas_solve(global_reference);
  const std::vector<dtdma::Scalar> reference(global_reference.rhs().begin(),
                                              global_reference.rhs().end());
  check_values(system_result.solution, reference);
  constexpr std::array<std::size_t, 2> partition_counts{2, 4};
  check_virtual_solutions(system_diagonal, reference, partition_counts);
  check_virtual_solutions(fully_batched, reference, partition_counts);
}

TEST_CASE("Prepared operators are reusable for unrelated right hand sides") {
  constexpr std::size_t row_count = 8;
  constexpr std::size_t batch_size = 2;
  const auto exact_a = make_exact(row_count, batch_size);
  auto exact_b = exact_a;
  for (auto& value : exact_b) {
    value = 10.0F - 0.75F * value;
  }

  dtdma::SharedTridiagonalBatch shared_a(row_count, batch_size);
  dtdma::SharedTridiagonalBatch shared_b(row_count, batch_size);
  dtdma::TridiagonalBatch full_a(row_count, batch_size);
  dtdma::TridiagonalBatch full_b(row_count, batch_size);
  for (std::size_t row = 0; row < row_count; ++row) {
    const dtdma::Scalar lower = row == 0 ? 0.0F : -0.5F;
    const dtdma::Scalar diagonal = 4.0F + 0.1F * row;
    const dtdma::Scalar upper = row + 1 == row_count ? 0.0F : -0.25F;
    shared_a.lower(row) = shared_b.lower(row) = lower;
    shared_a.diagonal(row) = shared_b.diagonal(row) = diagonal;
    shared_a.upper(row) = shared_b.upper(row) = upper;
    for (std::size_t system = 0; system < batch_size; ++system) {
      full_a.lower(row, system) = full_b.lower(row, system) = lower;
      full_a.diagonal(row, system) = full_b.diagonal(row, system) = diagonal;
      full_a.upper(row, system) = full_b.upper(row, system) = upper;
    }
  }
  construct_rhs(shared_a, exact_a);
  construct_rhs(shared_b, exact_b);
  construct_rhs(full_a, exact_a);
  construct_rhs(full_b, exact_b);

  dtdma::SharedPreparedOperator shared_prepared(row_count, batch_size);
  dtdma::prepare_forward_coefficients(shared_a, shared_prepared);
  dtdma::prepare_backward_coefficients(shared_a, shared_prepared);
  const std::vector<dtdma::Scalar> shared_lower(
      shared_prepared.prepared_lower().begin(),
      shared_prepared.prepared_lower().end());
  const std::vector<dtdma::Scalar> shared_diagonal(
      shared_prepared.prepared_diagonal().begin(),
      shared_prepared.prepared_diagonal().end());
  const std::vector<dtdma::Scalar> shared_upper(
      shared_prepared.prepared_upper().begin(),
      shared_prepared.prepared_upper().end());
  auto shared_reference_a = materialize_fully_batched(shared_a);
  dtdma::batched_thomas_solve(shared_reference_a);
  const std::vector<dtdma::Scalar> shared_result_a =
      solve_with_prepared(shared_a, shared_prepared);
  check_values(shared_result_a, exact_a);
  check_values(shared_result_a,
               {shared_reference_a.rhs().begin(),
                shared_reference_a.rhs().end()});
  CHECK(std::equal(shared_lower.begin(), shared_lower.end(),
                   shared_prepared.prepared_lower().begin()));
  CHECK(std::equal(shared_diagonal.begin(), shared_diagonal.end(),
                   shared_prepared.prepared_diagonal().begin()));
  CHECK(std::equal(shared_upper.begin(), shared_upper.end(),
                   shared_prepared.prepared_upper().begin()));

  auto shared_reference_b = materialize_fully_batched(shared_b);
  dtdma::batched_thomas_solve(shared_reference_b);
  const std::vector<dtdma::Scalar> shared_result_b =
      solve_with_prepared(shared_b, shared_prepared);
  check_values(shared_result_b, exact_b);
  check_values(shared_result_b,
               {shared_reference_b.rhs().begin(),
                shared_reference_b.rhs().end()});
  CHECK(std::equal(shared_lower.begin(), shared_lower.end(),
                   shared_prepared.prepared_lower().begin()));
  CHECK(std::equal(shared_diagonal.begin(), shared_diagonal.end(),
                   shared_prepared.prepared_diagonal().begin()));
  CHECK(std::equal(shared_upper.begin(), shared_upper.end(),
                   shared_prepared.prepared_upper().begin()));

  dtdma::PreparedOperatorBatch full_prepared(row_count, batch_size);
  dtdma::prepare_forward_coefficients(full_a, full_prepared);
  dtdma::prepare_backward_coefficients(full_a, full_prepared);
  const std::vector<dtdma::Scalar> full_lower(
      full_prepared.prepared_lower().begin(),
      full_prepared.prepared_lower().end());
  const std::vector<dtdma::Scalar> full_diagonal(
      full_prepared.prepared_diagonal().begin(),
      full_prepared.prepared_diagonal().end());
  const std::vector<dtdma::Scalar> full_upper(
      full_prepared.prepared_upper().begin(),
      full_prepared.prepared_upper().end());
  dtdma::TridiagonalBatch full_reference_a = full_a;
  dtdma::batched_thomas_solve(full_reference_a);
  const std::vector<dtdma::Scalar> full_result_a =
      solve_with_prepared(full_a, full_prepared);
  check_values(full_result_a, exact_a);
  check_values(full_result_a,
               {full_reference_a.rhs().begin(), full_reference_a.rhs().end()});
  CHECK(std::equal(full_lower.begin(), full_lower.end(),
                   full_prepared.prepared_lower().begin()));
  CHECK(std::equal(full_diagonal.begin(), full_diagonal.end(),
                   full_prepared.prepared_diagonal().begin()));
  CHECK(std::equal(full_upper.begin(), full_upper.end(),
                   full_prepared.prepared_upper().begin()));

  dtdma::TridiagonalBatch full_reference_b = full_b;
  dtdma::batched_thomas_solve(full_reference_b);
  const std::vector<dtdma::Scalar> full_result_b =
      solve_with_prepared(full_b, full_prepared);
  check_values(full_result_b, exact_b);
  check_values(full_result_b,
               {full_reference_b.rhs().begin(), full_reference_b.rhs().end()});
  CHECK(std::equal(full_lower.begin(), full_lower.end(),
                   full_prepared.prepared_lower().begin()));
  CHECK(std::equal(full_diagonal.begin(), full_diagonal.end(),
                   full_prepared.prepared_diagonal().begin()));
  CHECK(std::equal(full_upper.begin(), full_upper.end(),
                   full_prepared.prepared_upper().begin()));
}
