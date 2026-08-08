#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/backward_rhs_reduction.hpp"
#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/prepare_operator.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/rhs_batch.hpp"
#include "dtdma/endpoint_batch.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/single_partition_reconstruction.hpp"
#include "dtdma/single_partition_reduced_system.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

TEST_CASE("A single-partition solve matches a hand-checkable system") {
  constexpr dtdma::Scalar tolerance = 2.0e-5F;
  dtdma::TridiagonalBatch original(5, 1);
  dtdma::RhsBatch input_rhs(5, 1);
  const std::array<dtdma::Scalar, 5> lower{0.0F, 1.0F, 1.0F, 1.0F,
                                           1.0F};
  const std::array<dtdma::Scalar, 5> diagonal{4.0F, 4.0F, 4.0F, 4.0F,
                                              4.0F};
  const std::array<dtdma::Scalar, 5> upper{1.0F, 1.0F, 1.0F, 1.0F,
                                           0.0F};
  const std::array<dtdma::Scalar, 5> exact_solution{1.0F, 2.0F, 3.0F,
                                                    4.0F, 5.0F};
  const std::array<dtdma::Scalar, 5> rhs{6.0F, 12.0F, 18.0F, 24.0F,
                                         24.0F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower[row];
    original.diagonal(row, 0) = diagonal[row];
    original.upper(row, 0) = upper[row];
    input_rhs.rhs(row, 0) = rhs[row];
  }

  const std::vector<dtdma::Scalar> original_lower(original.lower().begin(),
                                                   original.lower().end());
  const std::vector<dtdma::Scalar> original_diagonal(
      original.diagonal().begin(), original.diagonal().end());
  const std::vector<dtdma::Scalar> original_upper(original.upper().begin(),
                                                   original.upper().end());
  const std::vector<dtdma::Scalar> original_rhs(input_rhs.rhs().begin(),
                                                 input_rhs.rhs().end());

  dtdma::TridiagonalBatch global_reference = original;
  dtdma::RhsBatch global_reference_rhs = input_rhs;
  dtdma::batched_thomas_solve(global_reference, global_reference_rhs);

  dtdma::TridiagonalBatch preparation_original = original;
  dtdma::PreparedOperatorBatch prepared =
      dtdma::prepare_operator(std::move(preparation_original));
  const std::vector<dtdma::Scalar> prepared_lower(
      prepared.prepared_lower().begin(), prepared.prepared_lower().end());
  const std::vector<dtdma::Scalar> prepared_diagonal(
      prepared.prepared_diagonal().begin(),
      prepared.prepared_diagonal().end());
  const std::vector<dtdma::Scalar> prepared_upper(
      prepared.prepared_upper().begin(), prepared.prepared_upper().end());

  dtdma::RhsBatch working(5, 1);
  dtdma::initialize_reduced_rhs(input_rhs, working);
  dtdma::reduce_rhs_forward(prepared, working);
  dtdma::reduce_rhs_backward(prepared, working);

  dtdma::EndpointBatch endpoints(1);
  dtdma::extract_reduced_rhs_endpoints(working, endpoints);
  CHECK(endpoints.endpoint(0, 0) ==
        Catch::Approx(3.64285714F).margin(tolerance));
  CHECK(endpoints.endpoint(0, 1) ==
        Catch::Approx(18.6428571F).margin(tolerance));

  dtdma::TridiagonalBatch reduced_system(2, 1);
  dtdma::RhsBatch reduced_rhs(2, 1);
  dtdma::assemble_single_partition_reduced_system(
      prepared, working, endpoints, reduced_system, reduced_rhs);

  CHECK(reduced_system.lower(0, 0) == 0.0F);
  CHECK(reduced_system.diagonal(0, 0) ==
        Catch::Approx(3.73214286F).margin(tolerance));
  CHECK(reduced_system.upper(0, 0) ==
        Catch::Approx(-0.0178571429F).margin(tolerance));
  CHECK(reduced_rhs.rhs(0, 0) ==
        Catch::Approx(3.64285714F).margin(tolerance));
  CHECK(reduced_system.lower(1, 0) ==
        Catch::Approx(-0.0178571429F).margin(tolerance));
  CHECK(reduced_system.diagonal(1, 0) ==
        Catch::Approx(3.73214286F).margin(tolerance));
  CHECK(reduced_system.upper(1, 0) == 0.0F);
  CHECK(reduced_rhs.rhs(1, 0) ==
        Catch::Approx(18.6428571F).margin(tolerance));

  dtdma::batched_thomas_solve(reduced_system, reduced_rhs);
  dtdma::recover_single_partition_endpoints(reduced_system, reduced_rhs,
                                             endpoints);
  CHECK(endpoints.endpoint(0, 0) ==
        Catch::Approx(exact_solution[0]).margin(tolerance));
  CHECK(endpoints.endpoint(0, 1) ==
        Catch::Approx(exact_solution[4]).margin(tolerance));

  dtdma::reconstruct_single_partition(prepared, working, endpoints);

  dtdma::Scalar maximum_difference = 0.0F;
  for (std::size_t row = 0; row < original.row_count(); ++row) {
    CHECK(working.rhs(row, 0) ==
          Catch::Approx(exact_solution[row]).margin(tolerance));
    CHECK(working.rhs(row, 0) ==
          Catch::Approx(global_reference_rhs.rhs(row, 0)).margin(tolerance));
    maximum_difference =
        std::max(maximum_difference,
                 std::abs(working.rhs(row, 0) -
                          global_reference_rhs.rhs(row, 0)));
  }
  CHECK(maximum_difference <= tolerance);

  CHECK(std::vector<dtdma::Scalar>(original.lower().begin(),
                                   original.lower().end()) == original_lower);
  CHECK(std::vector<dtdma::Scalar>(original.diagonal().begin(),
                                   original.diagonal().end()) ==
        original_diagonal);
  CHECK(std::vector<dtdma::Scalar>(original.upper().begin(),
                                   original.upper().end()) == original_upper);
  CHECK(std::vector<dtdma::Scalar>(input_rhs.rhs().begin(),
                                   input_rhs.rhs().end()) == original_rhs);
  CHECK(std::vector<dtdma::Scalar>(prepared.prepared_lower().begin(),
                                   prepared.prepared_lower().end()) ==
        prepared_lower);
  CHECK(std::vector<dtdma::Scalar>(prepared.prepared_diagonal().begin(),
                                   prepared.prepared_diagonal().end()) ==
        prepared_diagonal);
  CHECK(std::vector<dtdma::Scalar>(prepared.prepared_upper().begin(),
                                   prepared.prepared_upper().end()) ==
        prepared_upper);
}

TEST_CASE("Single-partition assembly canonicalizes physical exterior slots") {
  dtdma::TridiagonalBatch original(5, 1);
  dtdma::PreparedOperatorBatch prepared(5, 1);
  dtdma::RhsBatch working(5, 1);
  dtdma::EndpointBatch endpoints(1);
  dtdma::TridiagonalBatch reduced_system(2, 1);
  dtdma::RhsBatch reduced_rhs(2, 1);

  prepared.prepared_lower(0, 0) = 2.0F;
  prepared.prepared_diagonal(0, 0) = 10.0F;
  prepared.prepared_upper(0, 0) = 3.0F;
  prepared.prepared_lower(4, 0) = 4.0F;
  prepared.prepared_diagonal(4, 0) = 20.0F;
  prepared.prepared_upper(4, 0) = 5.0F;
  working.rhs(0, 0) = 7.0F;
  working.rhs(4, 0) = 9.0F;
  endpoints.endpoint(0, 0) = 7.0F;
  endpoints.endpoint(0, 1) = 9.0F;

  const std::vector<dtdma::Scalar> preserved_prepared_lower(
      prepared.prepared_lower().begin(), prepared.prepared_lower().end());
  const std::vector<dtdma::Scalar> preserved_prepared_diagonal(
      prepared.prepared_diagonal().begin(),
      prepared.prepared_diagonal().end());
  const std::vector<dtdma::Scalar> preserved_prepared_upper(
      prepared.prepared_upper().begin(), prepared.prepared_upper().end());
  const std::vector<dtdma::Scalar> preserved_working(working.rhs().begin(),
                                                     working.rhs().end());
  const std::vector<dtdma::Scalar> preserved_endpoints(
      endpoints.endpoints().begin(), endpoints.endpoints().end());

  dtdma::assemble_single_partition_reduced_system(
      prepared, working, endpoints, reduced_system, reduced_rhs);

  CHECK(reduced_system.lower(0, 0) == 0.0F);
  CHECK(reduced_system.diagonal(0, 0) == 10.0F);
  CHECK(reduced_system.upper(0, 0) == 3.0F);
  CHECK(reduced_rhs.rhs(0, 0) == 7.0F);
  CHECK(reduced_system.lower(1, 0) == 4.0F);
  CHECK(reduced_system.diagonal(1, 0) == 20.0F);
  CHECK(reduced_system.upper(1, 0) == 0.0F);
  CHECK(reduced_rhs.rhs(1, 0) == 9.0F);
  CHECK(std::vector<dtdma::Scalar>(prepared.prepared_lower().begin(),
                                   prepared.prepared_lower().end()) ==
        preserved_prepared_lower);
  CHECK(std::vector<dtdma::Scalar>(prepared.prepared_diagonal().begin(),
                                   prepared.prepared_diagonal().end()) ==
        preserved_prepared_diagonal);
  CHECK(std::vector<dtdma::Scalar>(prepared.prepared_upper().begin(),
                                   prepared.prepared_upper().end()) ==
        preserved_prepared_upper);
  CHECK(std::vector<dtdma::Scalar>(working.rhs().begin(),
                                   working.rhs().end()) == preserved_working);
  CHECK(std::vector<dtdma::Scalar>(endpoints.endpoints().begin(),
                                   endpoints.endpoints().end()) ==
        preserved_endpoints);
}

TEST_CASE("Single-partition reconstruction matches fixed interior values") {
  dtdma::PreparedOperatorBatch prepared(5, 1);
  dtdma::RhsBatch working(5, 1);
  dtdma::EndpointBatch solved_endpoints(1);
  const std::array<dtdma::Scalar, 5> reduced_rhs{99.0F, 7.0F, 10.0F,
                                                 4.0F, 88.0F};

  prepared.prepared_lower(1, 0) = 1.0F;
  prepared.prepared_diagonal(1, 0) = 2.0F;
  prepared.prepared_upper(1, 0) = 0.5F;
  prepared.prepared_lower(2, 0) = -2.0F;
  prepared.prepared_diagonal(2, 0) = 4.0F;
  prepared.prepared_upper(2, 0) = 1.0F;
  prepared.prepared_lower(3, 0) = 3.0F;
  prepared.prepared_diagonal(3, 0) = 5.0F;
  prepared.prepared_upper(3, 0) = -1.0F;
  for (std::size_t row = 0; row < working.row_count(); ++row) {
    working.rhs(row, 0) = reduced_rhs[row];
  }
  solved_endpoints.endpoint(0, 0) = 2.0F;
  solved_endpoints.endpoint(0, 1) = -3.0F;
  const std::vector<dtdma::Scalar> preserved_endpoints(
      solved_endpoints.endpoints().begin(), solved_endpoints.endpoints().end());

  dtdma::reconstruct_single_partition(prepared, working, solved_endpoints);

  CHECK(working.rhs(0, 0) == 2.0F);
  CHECK(working.rhs(1, 0) == 3.25F);
  CHECK(working.rhs(2, 0) == 4.25F);
  CHECK(working.rhs(3, 0) == -1.0F);
  CHECK(working.rhs(4, 0) == -3.0F);
  CHECK(std::vector<dtdma::Scalar>(solved_endpoints.endpoints().begin(),
                                   solved_endpoints.endpoints().end()) ==
        preserved_endpoints);
}

TEST_CASE("A row-varying single-partition solve matches global Thomas") {
  constexpr dtdma::Scalar tolerance = 2.0e-5F;
  dtdma::TridiagonalBatch original(6, 1);
  dtdma::RhsBatch input_rhs(6, 1);
  const std::array<dtdma::Scalar, 6> lower{0.0F, -0.5F, 1.25F, -1.0F,
                                           0.75F, 2.0F};
  const std::array<dtdma::Scalar, 6> diagonal{3.0F, 4.5F, 5.0F, 6.5F,
                                              7.0F, 8.0F};
  const std::array<dtdma::Scalar, 6> upper{1.0F, -0.75F, 0.5F, 1.25F,
                                           -1.0F, 0.0F};
  const std::array<dtdma::Scalar, 6> exact_solution{-2.0F, 0.5F, 3.0F,
                                                    -1.0F, 2.0F, 4.0F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower[row];
    original.diagonal(row, 0) = diagonal[row];
    original.upper(row, 0) = upper[row];
    dtdma::Scalar value = diagonal[row] * exact_solution[row];
    if (row > 0) {
      value += lower[row] * exact_solution[row - 1];
    }
    if (row + 1 < original.row_count()) {
      value += upper[row] * exact_solution[row + 1];
    }
    input_rhs.rhs(row, 0) = value;
  }

  dtdma::TridiagonalBatch global_reference = original;
  dtdma::RhsBatch global_reference_rhs = input_rhs;
  dtdma::batched_thomas_solve(global_reference, global_reference_rhs);
  dtdma::TridiagonalBatch preparation_original = original;
  dtdma::PreparedOperatorBatch prepared =
      dtdma::prepare_operator(std::move(preparation_original));
  dtdma::RhsBatch working(6, 1);
  dtdma::initialize_reduced_rhs(input_rhs, working);
  dtdma::reduce_rhs_forward(prepared, working);
  dtdma::reduce_rhs_backward(prepared, working);
  dtdma::EndpointBatch endpoints(1);
  dtdma::extract_reduced_rhs_endpoints(working, endpoints);
  dtdma::TridiagonalBatch reduced_system(2, 1);
  dtdma::RhsBatch reduced_rhs(2, 1);
  dtdma::assemble_single_partition_reduced_system(
      prepared, working, endpoints, reduced_system, reduced_rhs);
  dtdma::batched_thomas_solve(reduced_system, reduced_rhs);
  dtdma::recover_single_partition_endpoints(reduced_system, reduced_rhs,
                                             endpoints);
  dtdma::reconstruct_single_partition(prepared, working, endpoints);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    CHECK(working.rhs(row, 0) ==
          Catch::Approx(exact_solution[row]).margin(tolerance));
    CHECK(working.rhs(row, 0) ==
          Catch::Approx(global_reference_rhs.rhs(row, 0)).margin(tolerance));
  }
}

TEST_CASE("A batched single-partition solve keeps systems independent") {
  constexpr std::size_t row_count = 5;
  constexpr std::size_t batch_size = 3;
  constexpr dtdma::Scalar tolerance = 2.0e-5F;
  dtdma::TridiagonalBatch original(row_count, batch_size);
  dtdma::RhsBatch input_rhs(row_count, batch_size);
  const std::array<dtdma::Scalar, row_count * batch_size> exact_solution{
      1.0F, -1.0F, 3.0F,
      2.0F, 0.5F, -2.0F,
      3.0F, 2.0F, 0.75F,
      4.0F, -3.0F, 1.5F,
      5.0F, 1.25F, -1.0F};
  const std::array<dtdma::Scalar, row_count * batch_size> lower{
      0.0F, 0.0F, 0.0F,
      1.0F, -1.0F, 0.25F,
      1.0F, 0.5F, -1.5F,
      1.0F, 1.5F, 0.5F,
      1.0F, -0.75F, 2.0F};
  const std::array<dtdma::Scalar, row_count * batch_size> diagonal{
      4.0F, 5.0F, 4.5F,
      4.0F, 6.0F, 5.5F,
      4.0F, 7.0F, 6.5F,
      4.0F, 8.0F, 7.5F,
      4.0F, 9.0F, 8.5F};
  const std::array<dtdma::Scalar, row_count * batch_size> upper{
      1.0F, -0.5F, 1.25F,
      1.0F, 1.0F, -0.5F,
      1.0F, 0.75F, 1.5F,
      1.0F, -1.25F, 0.25F,
      0.0F, 0.0F, 0.0F};

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const std::size_t index = row * batch_size + system;
      original.lower(row, system) = lower[index];
      original.diagonal(row, system) = diagonal[index];
      original.upper(row, system) = upper[index];
      dtdma::Scalar value = diagonal[index] * exact_solution[index];
      if (row > 0) {
        value += lower[index] *
                 exact_solution[(row - 1) * batch_size + system];
      }
      if (row + 1 < row_count) {
        value += upper[index] *
                 exact_solution[(row + 1) * batch_size + system];
      }
      input_rhs.rhs(row, system) = value;
    }
  }

  dtdma::TridiagonalBatch global_reference = original;
  dtdma::RhsBatch global_reference_rhs = input_rhs;
  dtdma::batched_thomas_solve(global_reference, global_reference_rhs);
  dtdma::TridiagonalBatch preparation_original = original;
  dtdma::PreparedOperatorBatch prepared =
      dtdma::prepare_operator(std::move(preparation_original));
  dtdma::RhsBatch working(row_count, batch_size);
  dtdma::initialize_reduced_rhs(input_rhs, working);
  dtdma::reduce_rhs_forward(prepared, working);
  dtdma::reduce_rhs_backward(prepared, working);
  dtdma::EndpointBatch endpoints(batch_size);
  dtdma::extract_reduced_rhs_endpoints(working, endpoints);
  dtdma::TridiagonalBatch reduced_system(2, batch_size);
  dtdma::RhsBatch reduced_rhs(2, batch_size);
  dtdma::assemble_single_partition_reduced_system(
      prepared, working, endpoints, reduced_system, reduced_rhs);
  dtdma::batched_thomas_solve(reduced_system, reduced_rhs);
  dtdma::recover_single_partition_endpoints(reduced_system, reduced_rhs,
                                             endpoints);
  dtdma::reconstruct_single_partition(prepared, working, endpoints);

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const std::size_t index = row * batch_size + system;
      CHECK(working.rhs(row, system) ==
            Catch::Approx(exact_solution[index]).margin(tolerance));
      CHECK(working.rhs(row, system) ==
            Catch::Approx(global_reference_rhs.rhs(row, system))
                .margin(tolerance));
    }
  }
}

TEST_CASE("Single-partition assembly rejects incompatible dimensions") {
  const dtdma::TridiagonalBatch too_short(2, 1);
  const dtdma::PreparedOperatorBatch too_short_prepared(2, 1);
  const dtdma::RhsBatch too_short_working(2, 1);
  const dtdma::EndpointBatch one_endpoint_batch(1);
  dtdma::TridiagonalBatch reduced_one(2, 1);
  dtdma::RhsBatch reduced_one_rhs(2, 1);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      too_short_prepared, too_short_working,
                      one_endpoint_batch, reduced_one, reduced_one_rhs),
                  std::invalid_argument);

  const dtdma::TridiagonalBatch original(5, 2);
  const dtdma::PreparedOperatorBatch prepared(5, 2);
  const dtdma::RhsBatch working(5, 2);
  const dtdma::EndpointBatch endpoints(2);
  const dtdma::PreparedOperatorBatch wrong_prepared_rows(4, 2);
  const dtdma::PreparedOperatorBatch wrong_prepared_batch(5, 3);
  const dtdma::RhsBatch wrong_working_rows(4, 2);
  const dtdma::RhsBatch wrong_working_batch(5, 3);
  const dtdma::EndpointBatch wrong_endpoints(3);
  dtdma::TridiagonalBatch reduced(2, 2);
  dtdma::RhsBatch reduced_rhs(2, 2);
  dtdma::TridiagonalBatch wrong_reduced_rows(3, 2);
  dtdma::TridiagonalBatch wrong_reduced_batch(2, 3);

  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      wrong_prepared_rows, working, endpoints,
                      reduced, reduced_rhs),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      wrong_prepared_batch, working, endpoints,
                      reduced, reduced_rhs),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      prepared, wrong_working_rows, endpoints,
                      reduced, reduced_rhs),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      prepared, wrong_working_batch, endpoints,
                      reduced, reduced_rhs),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      prepared, working, wrong_endpoints, reduced,
                      reduced_rhs),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      prepared, working, endpoints,
                      wrong_reduced_rows, reduced_rhs),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::assemble_single_partition_reduced_system(
                      prepared, working, endpoints,
                      wrong_reduced_batch, reduced_rhs),
                  std::invalid_argument);
}

TEST_CASE("Endpoint recovery and reconstruction reject incompatible dimensions") {
  const dtdma::TridiagonalBatch wrong_reduced_rows(3, 2);
  const dtdma::TridiagonalBatch wrong_reduced_batch(2, 3);
  const dtdma::RhsBatch reduced_rhs(2, 2);
  dtdma::EndpointBatch endpoints(2);
  CHECK_THROWS_AS(dtdma::recover_single_partition_endpoints(
                      wrong_reduced_rows, reduced_rhs, endpoints),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::recover_single_partition_endpoints(
                      wrong_reduced_batch, reduced_rhs, endpoints),
                  std::invalid_argument);

  const dtdma::PreparedOperatorBatch too_short_prepared(2, 1);
  dtdma::RhsBatch too_short_working(2, 1);
  const dtdma::EndpointBatch one_endpoint_batch(1);
  CHECK_THROWS_AS(dtdma::reconstruct_single_partition(
                      too_short_prepared, too_short_working,
                      one_endpoint_batch),
                  std::invalid_argument);

  const dtdma::PreparedOperatorBatch prepared(5, 2);
  dtdma::RhsBatch wrong_working_rows(4, 2);
  dtdma::RhsBatch wrong_working_batch(5, 3);
  const dtdma::EndpointBatch wrong_endpoints(3);
  CHECK_THROWS_AS(dtdma::reconstruct_single_partition(
                      prepared, wrong_working_rows, endpoints),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::reconstruct_single_partition(
                      prepared, wrong_working_batch, endpoints),
                  std::invalid_argument);
  dtdma::RhsBatch working(5, 2);
  CHECK_THROWS_AS(dtdma::reconstruct_single_partition(
                      prepared, working, wrong_endpoints),
                  std::invalid_argument);
}
