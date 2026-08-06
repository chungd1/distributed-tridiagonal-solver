#include "dtdma/forward_rhs_reduction.hpp"
#include "dtdma/indexing.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

TEST_CASE("A reduced RHS batch mirrors the canonical batch layout") {
  dtdma::ReducedRhsBatch working(4, 3);

  CHECK(working.row_count() == 4);
  CHECK(working.batch_size() == 3);
  CHECK(working.element_count() == 12);
  CHECK(working.rhs().size() == working.element_count());

  working.rhs(2, 1) = 7.0F;
  CHECK(working.rhs()[dtdma::canonical_index(2, 1, 3)] == 7.0F);
  CHECK(&working.rhs(1, 1) == &working.rhs(1, 0) + 1);

  const dtdma::ReducedRhsBatch& const_working = working;
  STATIC_CHECK(std::is_same_v<decltype(working.rhs()),
                              std::span<dtdma::Scalar>>);
  STATIC_CHECK(std::is_same_v<decltype(const_working.rhs()),
                              std::span<const dtdma::Scalar>>);
  CHECK(const_working.rhs(2, 1) == 7.0F);

  CHECK_THROWS_AS(working.rhs(4, 0), std::out_of_range);
  CHECK_THROWS_AS(const_working.rhs(0, 3), std::out_of_range);
}

TEST_CASE("A reduced RHS batch rejects invalid dimensions") {
  CHECK_THROWS_AS(dtdma::ReducedRhsBatch(0, 2), std::invalid_argument);
  CHECK_THROWS_AS(dtdma::ReducedRhsBatch(3, 0), std::invalid_argument);

  const auto maximum = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(dtdma::ReducedRhsBatch(maximum, 2), std::length_error);
}

TEST_CASE("Reduced RHS initialization copies every canonical entry") {
  dtdma::TridiagonalBatch original(3, 2);
  dtdma::ReducedRhsBatch working(3, 2);
  const std::array<dtdma::Scalar, 6> rhs{1.0F, 10.0F, 2.0F,
                                         20.0F, 3.0F, 30.0F};

  for (std::size_t index = 0; index < original.element_count(); ++index) {
    original.rhs()[index] = rhs[index];
    working.rhs()[index] = -1.0F;
  }

  dtdma::initialize_reduced_rhs(original, working);

  for (std::size_t index = 0; index < original.element_count(); ++index) {
    CHECK(working.rhs()[index] == rhs[index]);
    CHECK(original.rhs()[index] == rhs[index]);
  }
}

TEST_CASE("Forward RHS reduction matches a complete hand calculation") {
  dtdma::TridiagonalBatch original(5, 1);
  dtdma::PreparedOperatorBatch prepared(5, 1);
  dtdma::ReducedRhsBatch working(5, 1);
  const std::array<dtdma::Scalar, 5> lower{9.0F, 2.0F, 3.0F, 4.0F,
                                           5.0F};
  const std::array<dtdma::Scalar, 5> prepared_diagonal{
      8.0F, 10.0F, 20.0F, 25.0F, 30.0F};
  const std::array<dtdma::Scalar, 5> rhs{1.0F, 2.0F, 10.0F, 20.0F,
                                         30.0F};
  const std::array<dtdma::Scalar, 5> expected_rhs{
      1.0F, 2.0F, 9.4F, 18.12F, 26.376F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower[row];
    original.rhs(row, 0) = rhs[row];
    prepared.prepared_diagonal(row, 0) = prepared_diagonal[row];
  }

  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);

  CHECK(working.rhs(0, 0) == expected_rhs[0]);
  CHECK(working.rhs(1, 0) == expected_rhs[1]);
  CHECK(working.rhs(2, 0) ==
        Catch::Approx(expected_rhs[2]).margin(1.0e-5F));
  CHECK(working.rhs(3, 0) ==
        Catch::Approx(expected_rhs[3]).margin(1.0e-5F));
  CHECK(working.rhs(4, 0) ==
        Catch::Approx(expected_rhs[4]).margin(1.0e-5F));
}

TEST_CASE("Forward RHS reduction keeps varying batch systems independent") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);
  dtdma::ReducedRhsBatch working(4, 2);

  const std::array<dtdma::Scalar, 4> lower_0{9.0F, 2.0F, 3.0F, 4.0F};
  const std::array<dtdma::Scalar, 4> diagonal_0{8.0F, 10.0F, 20.0F,
                                                25.0F};
  const std::array<dtdma::Scalar, 4> rhs_0{1.0F, 2.0F, 10.0F, 20.0F};
  const std::array<dtdma::Scalar, 4> lower_1{-7.0F, 5.0F, -2.0F,
                                             3.0F};
  const std::array<dtdma::Scalar, 4> diagonal_1{6.0F, 8.0F, 12.0F,
                                                14.0F};
  const std::array<dtdma::Scalar, 4> rhs_1{100.0F, -4.0F, 7.0F, 5.0F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower_0[row];
    original.rhs(row, 0) = rhs_0[row];
    prepared.prepared_diagonal(row, 0) = diagonal_0[row];
    original.lower(row, 1) = lower_1[row];
    original.rhs(row, 1) = rhs_1[row];
    prepared.prepared_diagonal(row, 1) = diagonal_1[row];
  }

  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);

  CHECK(working.rhs(0, 0) == 1.0F);
  CHECK(working.rhs(1, 0) == 2.0F);
  CHECK(working.rhs(2, 0) == Catch::Approx(9.4F));
  CHECK(working.rhs(3, 0) == Catch::Approx(18.12F));
  CHECK(working.rhs(0, 1) == 100.0F);
  CHECK(working.rhs(1, 1) == -4.0F);
  CHECK(working.rhs(2, 1) == Catch::Approx(6.0F));
  CHECK(working.rhs(3, 1) == Catch::Approx(3.5F));
}

TEST_CASE("Forward RHS reduction supports the minimum row count") {
  dtdma::TridiagonalBatch original(3, 1);
  dtdma::PreparedOperatorBatch prepared(3, 1);
  dtdma::ReducedRhsBatch working(3, 1);
  original.lower(2, 0) = 4.0F;
  original.rhs(0, 0) = 3.0F;
  original.rhs(1, 0) = 6.0F;
  original.rhs(2, 0) = 10.0F;
  prepared.prepared_diagonal(1, 0) = 8.0F;

  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);

  CHECK(working.rhs(0, 0) == 3.0F);
  CHECK(working.rhs(1, 0) == 6.0F);
  CHECK(working.rhs(2, 0) == 7.0F);
}

TEST_CASE("Forward RHS reduction rejects unsupported or incompatible batches") {
  const dtdma::TridiagonalBatch too_short(2, 1);
  dtdma::PreparedOperatorBatch too_short_prepared(2, 1);
  dtdma::ReducedRhsBatch too_short_working(2, 1);
  CHECK_THROWS_AS(
      dtdma::initialize_reduced_rhs(too_short, too_short_working),
      std::invalid_argument);
  CHECK_THROWS_AS(dtdma::reduce_rhs_forward(
                      too_short, too_short_prepared, too_short_working),
                  std::invalid_argument);

  const dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);
  dtdma::PreparedOperatorBatch wrong_prepared_rows(3, 2);
  dtdma::PreparedOperatorBatch wrong_prepared_batch(4, 3);
  dtdma::ReducedRhsBatch working(4, 2);
  dtdma::ReducedRhsBatch wrong_working_rows(3, 2);
  dtdma::ReducedRhsBatch wrong_working_batch(4, 3);

  CHECK_THROWS_AS(
      dtdma::initialize_reduced_rhs(original, wrong_working_rows),
      std::invalid_argument);
  CHECK_THROWS_AS(
      dtdma::initialize_reduced_rhs(original, wrong_working_batch),
      std::invalid_argument);
  CHECK_THROWS_AS(dtdma::reduce_rhs_forward(
                      original, wrong_prepared_rows, working),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::reduce_rhs_forward(
                      original, wrong_prepared_batch, working),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::reduce_rhs_forward(
                      original, prepared, wrong_working_rows),
                  std::invalid_argument);
  CHECK_THROWS_AS(dtdma::reduce_rhs_forward(
                      original, prepared, wrong_working_batch),
                  std::invalid_argument);
}

TEST_CASE("Forward RHS reduction preserves the original and prepared batches") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);
  dtdma::ReducedRhsBatch working(4, 2);

  for (std::size_t index = 0; index < original.element_count(); ++index) {
    const auto value = static_cast<dtdma::Scalar>(index + 1);
    original.lower()[index] = 0.1F * value;
    original.diagonal()[index] = 10.0F + value;
    original.upper()[index] = 0.2F * value;
    original.rhs()[index] = 20.0F + value;
    prepared.prepared_lower()[index] = -0.3F * value;
    prepared.prepared_diagonal()[index] = 12.0F + value;
    prepared.prepared_upper()[index] = 0.4F * value;
  }

  const std::vector<dtdma::Scalar> original_lower(original.lower().begin(),
                                                   original.lower().end());
  const std::vector<dtdma::Scalar> original_diagonal(
      original.diagonal().begin(), original.diagonal().end());
  const std::vector<dtdma::Scalar> original_upper(original.upper().begin(),
                                                   original.upper().end());
  const std::vector<dtdma::Scalar> original_rhs(original.rhs().begin(),
                                                 original.rhs().end());
  const std::vector<dtdma::Scalar> prepared_lower(
      prepared.prepared_lower().begin(), prepared.prepared_lower().end());
  const std::vector<dtdma::Scalar> prepared_diagonal(
      prepared.prepared_diagonal().begin(),
      prepared.prepared_diagonal().end());
  const std::vector<dtdma::Scalar> prepared_upper(
      prepared.prepared_upper().begin(), prepared.prepared_upper().end());

  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);

  CHECK(std::vector<dtdma::Scalar>(original.lower().begin(),
                                   original.lower().end()) == original_lower);
  CHECK(std::vector<dtdma::Scalar>(original.diagonal().begin(),
                                   original.diagonal().end()) ==
        original_diagonal);
  CHECK(std::vector<dtdma::Scalar>(original.upper().begin(),
                                   original.upper().end()) == original_upper);
  CHECK(std::vector<dtdma::Scalar>(original.rhs().begin(),
                                   original.rhs().end()) == original_rhs);
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

TEST_CASE("Forward RHS reduction agrees with the active CaNS reference fixture") {
  // CaNS-World/CaNS main, 245a23348ef795af9aebeda6c767a46ca8be45e8,
  // src/solver_gpu.f90::gaussel_dtdma_gpu_fast_1d; reciprocal-diagonal
  // multiplication ordering with norm = 1.
  constexpr dtdma::Scalar cans_tolerance = 1.0e-6F;
  const std::array<dtdma::Scalar, 5> original_lower{
      0.75F, -1.25F, 0.875F, -1.5F, 1.125F};
  const std::array<dtdma::Scalar, 5> original_diagonal{
      4.5F, 5.25F, 6.125F, 5.875F, 6.75F};
  const std::array<dtdma::Scalar, 5> original_upper{
      -0.625F, 1.375F, -0.875F, 1.25F, -0.5F};
  const std::array<dtdma::Scalar, 5> prepared_diagonal{
      4.3451786F, 5.25F, 5.89583349F, 5.65238523F, 6.50121117F};
  const std::array<dtdma::Scalar, 5> original_rhs{
      2.0F, -3.0F, 4.5F, -6.0F, 7.25F};
  const std::array<dtdma::Scalar, 5> cans_reduced_rhs{
      2.0F, -3.0F, 5.0F, -4.72791529F, 8.19100189F};
  dtdma::TridiagonalBatch original(5, 1);
  dtdma::PreparedOperatorBatch prepared(5, 1);
  dtdma::ReducedRhsBatch working(5, 1);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = original_lower[row];
    original.diagonal(row, 0) = original_diagonal[row];
    original.upper(row, 0) = original_upper[row];
    original.rhs(row, 0) = original_rhs[row];
    prepared.prepared_diagonal(row, 0) = prepared_diagonal[row];
  }

  dtdma::initialize_reduced_rhs(original, working);
  dtdma::reduce_rhs_forward(original, prepared, working);

  CHECK(working.rhs(0, 0) ==
        Catch::Approx(cans_reduced_rhs[0]).margin(cans_tolerance));
  CHECK(working.rhs(1, 0) ==
        Catch::Approx(cans_reduced_rhs[1]).margin(cans_tolerance));
  CHECK(working.rhs(2, 0) ==
        Catch::Approx(cans_reduced_rhs[2]).margin(cans_tolerance));
  CHECK(working.rhs(3, 0) ==
        Catch::Approx(cans_reduced_rhs[3]).margin(cans_tolerance));
  CHECK(working.rhs(4, 0) ==
        Catch::Approx(cans_reduced_rhs[4]).margin(cans_tolerance));

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    CHECK(original.lower(row, 0) == original_lower[row]);
    CHECK(original.diagonal(row, 0) == original_diagonal[row]);
    CHECK(original.upper(row, 0) == original_upper[row]);
    CHECK(original.rhs(row, 0) == original_rhs[row]);
    CHECK(prepared.prepared_diagonal(row, 0) == prepared_diagonal[row]);
  }
}
