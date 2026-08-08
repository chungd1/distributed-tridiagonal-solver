#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/prepared_operator_batch.hpp"
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

TEST_CASE("A prepared operator batch mirrors the original batch layout") {
  const dtdma::TridiagonalBatch original(4, 3);
  dtdma::PreparedOperatorBatch prepared(4, 3);

  CHECK(prepared.row_count() == original.row_count());
  CHECK(prepared.batch_size() == original.system_count());
  CHECK(prepared.element_count() == original.element_count());
  CHECK(prepared.prepared_lower().size() == original.element_count());
  CHECK(prepared.prepared_diagonal().size() == original.element_count());
  CHECK(prepared.prepared_upper().size() == original.element_count());

  prepared.prepared_diagonal(2, 1) = 7.0F;
  CHECK(prepared.prepared_diagonal()[
            dtdma::canonical_index(2, 1, prepared.batch_size())] == 7.0F);

  const dtdma::PreparedOperatorBatch& const_prepared = prepared;
  STATIC_CHECK(std::is_same_v<decltype(const_prepared.prepared_lower()),
                              std::span<const dtdma::Scalar>>);
  STATIC_CHECK(std::is_same_v<decltype(prepared.prepared_lower()),
                              std::span<dtdma::Scalar>>);
}

TEST_CASE("A prepared operator batch rejects invalid dimensions") {
  CHECK_THROWS_AS(dtdma::PreparedOperatorBatch(0, 2), std::invalid_argument);
  CHECK_THROWS_AS(dtdma::PreparedOperatorBatch(3, 0), std::invalid_argument);

  const auto maximum = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(dtdma::PreparedOperatorBatch(maximum, 2),
                  std::length_error);
}

TEST_CASE("Forward coefficient preparation matches a hand calculation") {
  dtdma::TridiagonalBatch original(4, 1);
  dtdma::PreparedOperatorBatch prepared(4, 1);

  original.lower(0, 0) = 9.0F;
  original.lower(1, 0) = 2.0F;
  original.lower(2, 0) = 3.0F;
  original.lower(3, 0) = 4.0F;
  original.diagonal(0, 0) = 9.0F;
  original.diagonal(1, 0) = 10.0F;
  original.diagonal(2, 0) = 20.0F;
  original.diagonal(3, 0) = 30.0F;
  original.upper(0, 0) = 5.0F;
  original.upper(1, 0) = 6.0F;
  original.upper(2, 0) = 7.0F;
  original.upper(3, 0) = 8.0F;

  dtdma::prepare_forward_coefficients(original, prepared);

  CHECK(prepared.prepared_lower(1, 0) == 2.0F);
  CHECK(prepared.prepared_diagonal(1, 0) == 10.0F);
  CHECK(prepared.prepared_diagonal(2, 0) == Catch::Approx(18.2F));
  CHECK(prepared.prepared_lower(2, 0) == Catch::Approx(-0.6F));
  CHECK(prepared.prepared_diagonal(3, 0) ==
        Catch::Approx(28.461538F));
  CHECK(prepared.prepared_lower(3, 0) == Catch::Approx(0.13186813F));
}

TEST_CASE("Forward coefficient preparation keeps batch systems independent") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);

  original.lower(1, 0) = 2.0F;
  original.diagonal(1, 0) = 10.0F;
  original.upper(1, 0) = 6.0F;
  original.lower(2, 0) = 3.0F;
  original.diagonal(2, 0) = 20.0F;
  original.upper(2, 0) = 7.0F;
  original.lower(3, 0) = 4.0F;
  original.diagonal(3, 0) = 30.0F;

  original.lower(1, 1) = -2.0F;
  original.diagonal(1, 1) = 8.0F;
  original.upper(1, 1) = 3.0F;
  original.lower(2, 1) = 4.0F;
  original.diagonal(2, 1) = 14.0F;
  original.upper(2, 1) = -2.0F;
  original.lower(3, 1) = -3.0F;
  original.diagonal(3, 1) = 12.0F;

  dtdma::prepare_forward_coefficients(original, prepared);

  CHECK(prepared.prepared_diagonal(2, 0) == Catch::Approx(18.2F));
  CHECK(prepared.prepared_lower(2, 0) == Catch::Approx(-0.6F));
  CHECK(prepared.prepared_diagonal(3, 0) ==
        Catch::Approx(28.461538F));
  CHECK(prepared.prepared_lower(3, 0) == Catch::Approx(0.13186813F));

  CHECK(prepared.prepared_diagonal(2, 1) == Catch::Approx(12.5F));
  CHECK(prepared.prepared_lower(2, 1) == Catch::Approx(1.0F));
  CHECK(prepared.prepared_diagonal(3, 1) == Catch::Approx(11.52F));
  CHECK(prepared.prepared_lower(3, 1) == Catch::Approx(0.24F));
}

TEST_CASE("Forward coefficient preparation preserves original batch data") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    for (std::size_t system = 0; system < original.system_count(); ++system) {
      const auto value = static_cast<dtdma::Scalar>(row * 3 + system + 1);
      original.lower(row, system) = value;
      original.diagonal(row, system) = value + 10.0F;
      original.upper(row, system) = value + 2.0F;
    }
  }

  const std::vector<dtdma::Scalar> lower(original.lower().begin(),
                                          original.lower().end());
  const std::vector<dtdma::Scalar> diagonal(original.diagonal().begin(),
                                             original.diagonal().end());
  const std::vector<dtdma::Scalar> upper(original.upper().begin(),
                                          original.upper().end());

  dtdma::prepare_forward_coefficients(original, prepared);

  CHECK(std::vector<dtdma::Scalar>(original.lower().begin(),
                                   original.lower().end()) == lower);
  CHECK(std::vector<dtdma::Scalar>(original.diagonal().begin(),
                                   original.diagonal().end()) == diagonal);
  CHECK(std::vector<dtdma::Scalar>(original.upper().begin(),
                                   original.upper().end()) == upper);
}

TEST_CASE("Forward coefficient preparation leaves other entries initialized") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);

  for (std::size_t system = 0; system < original.system_count(); ++system) {
    original.lower(1, system) = 1.0F;
    original.diagonal(1, system) = 4.0F;
    original.upper(1, system) = 1.0F;
    original.lower(2, system) = 1.0F;
    original.diagonal(2, system) = 4.0F;
    original.upper(2, system) = 1.0F;
    original.lower(3, system) = 1.0F;
    original.diagonal(3, system) = 4.0F;
  }

  dtdma::prepare_forward_coefficients(original, prepared);

  for (std::size_t system = 0; system < original.system_count(); ++system) {
    CHECK(prepared.prepared_lower(0, system) == 0.0F);
    CHECK(prepared.prepared_diagonal(0, system) == 0.0F);
  }
  for (const auto value : prepared.prepared_upper()) {
    CHECK(value == 0.0F);
  }
}

TEST_CASE("Forward coefficient preparation supports the minimum row count") {
  dtdma::TridiagonalBatch original(3, 1);
  dtdma::PreparedOperatorBatch prepared(3, 1);
  original.lower(1, 0) = 2.0F;
  original.diagonal(1, 0) = 8.0F;
  original.upper(1, 0) = 3.0F;
  original.lower(2, 0) = 4.0F;
  original.diagonal(2, 0) = 14.0F;

  dtdma::prepare_forward_coefficients(original, prepared);

  CHECK(prepared.prepared_lower(1, 0) == 2.0F);
  CHECK(prepared.prepared_diagonal(1, 0) == 8.0F);
  CHECK(prepared.prepared_lower(2, 0) == Catch::Approx(-1.0F));
  CHECK(prepared.prepared_diagonal(2, 0) == Catch::Approx(12.5F));

  dtdma::TridiagonalBatch too_short(2, 1);
  dtdma::PreparedOperatorBatch too_short_prepared(2, 1);
  CHECK_THROWS_AS(
      dtdma::prepare_forward_coefficients(too_short, too_short_prepared),
      std::invalid_argument);
}

TEST_CASE("Forward coefficient preparation rejects incompatible dimensions") {
  const dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch wrong_rows(3, 2);
  dtdma::PreparedOperatorBatch wrong_batch_size(4, 3);

  CHECK_THROWS_AS(
      dtdma::prepare_forward_coefficients(original, wrong_rows),
      std::invalid_argument);
  CHECK_THROWS_AS(
      dtdma::prepare_forward_coefficients(original, wrong_batch_size),
      std::invalid_argument);
}

TEST_CASE("Backward coefficient preparation matches complete hand calculations") {
  dtdma::TridiagonalBatch original(5, 1);
  dtdma::PreparedOperatorBatch prepared(5, 1);
  const std::array<dtdma::Scalar, 5> lower{2.0F, 1.0F, 2.0F, 3.0F,
                                           4.0F};
  const std::array<dtdma::Scalar, 5> diagonal{10.0F, 10.0F, 10.2F,
                                              10.3F, 10.4F};
  const std::array<dtdma::Scalar, 5> upper{1.0F, 1.0F, 1.0F, 1.0F,
                                           5.0F};
  const std::array<dtdma::Scalar, 5> expected_lower{
      2.0F, 1.0206F, -0.206F, 0.06F, -0.024F};
  const std::array<dtdma::Scalar, 5> expected_diagonal{
      9.89794F, 10.0F, 10.0F, 10.0F, 10.0F};
  const std::array<dtdma::Scalar, 5> expected_upper{
      -0.001F, 0.01F, -0.1F, 1.0F, 5.0F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower[row];
    original.diagonal(row, 0) = diagonal[row];
    original.upper(row, 0) = upper[row];
  }

  dtdma::prepare_forward_coefficients(original, prepared);
  dtdma::prepare_backward_coefficients(original, prepared);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    CHECK(prepared.prepared_lower(row, 0) ==
          Catch::Approx(expected_lower[row]).margin(1.0e-5F));
    CHECK(prepared.prepared_diagonal(row, 0) ==
          Catch::Approx(expected_diagonal[row]).margin(1.0e-5F));
    CHECK(prepared.prepared_upper(row, 0) ==
          Catch::Approx(expected_upper[row]).margin(1.0e-5F));
  }
}

TEST_CASE("Complete coefficient preparation keeps varying systems independent") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);

  const std::array<dtdma::Scalar, 4> lower_0{9.0F, 2.0F, 3.0F, 4.0F};
  const std::array<dtdma::Scalar, 4> diagonal_0{9.0F, 10.0F, 20.0F,
                                                30.0F};
  const std::array<dtdma::Scalar, 4> upper_0{5.0F, 6.0F, 7.0F, 8.0F};
  const std::array<dtdma::Scalar, 4> lower_1{-4.0F, -2.0F, 4.0F,
                                             -3.0F};
  const std::array<dtdma::Scalar, 4> diagonal_1{11.0F, 8.0F, 14.0F,
                                                12.0F};
  const std::array<dtdma::Scalar, 4> upper_1{2.0F, 3.0F, -2.0F, 5.0F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower_0[row];
    original.diagonal(row, 0) = diagonal_0[row];
    original.upper(row, 0) = upper_0[row];
    original.lower(row, 1) = lower_1[row];
    original.diagonal(row, 1) = diagonal_1[row];
    original.upper(row, 1) = upper_1[row];
  }

  dtdma::prepare_forward_coefficients(original, prepared);
  dtdma::prepare_backward_coefficients(original, prepared);

  CHECK(prepared.prepared_lower(1, 0) == Catch::Approx(2.1978022F));
  CHECK(prepared.prepared_diagonal(0, 0) == Catch::Approx(7.9010987F));
  CHECK(prepared.prepared_upper(1, 0) == Catch::Approx(-2.3076923F));

  CHECK(prepared.prepared_lower(0, 1) == -4.0F);
  CHECK(prepared.prepared_lower(1, 1) == Catch::Approx(-2.24F));
  CHECK(prepared.prepared_lower(2, 1) == Catch::Approx(1.0F));
  CHECK(prepared.prepared_lower(3, 1) == Catch::Approx(0.24F));
  CHECK(prepared.prepared_diagonal(0, 1) == Catch::Approx(11.56F));
  CHECK(prepared.prepared_diagonal(1, 1) == 8.0F);
  CHECK(prepared.prepared_diagonal(2, 1) == Catch::Approx(12.5F));
  CHECK(prepared.prepared_diagonal(3, 1) == Catch::Approx(11.52F));
  CHECK(prepared.prepared_upper(0, 1) == Catch::Approx(-0.12F));
  CHECK(prepared.prepared_upper(1, 1) == Catch::Approx(0.48F));
  CHECK(prepared.prepared_upper(2, 1) == -2.0F);
  CHECK(prepared.prepared_upper(3, 1) == 5.0F);
}

TEST_CASE("Complete coefficient preparation supports three-row systems") {
  dtdma::TridiagonalBatch original(3, 1);
  dtdma::PreparedOperatorBatch prepared(3, 1);
  const std::array<dtdma::Scalar, 3> lower{5.0F, 2.0F, 4.0F};
  const std::array<dtdma::Scalar, 3> diagonal{9.0F, 8.0F, 14.0F};
  const std::array<dtdma::Scalar, 3> upper{3.0F, 3.0F, 7.0F};

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = lower[row];
    original.diagonal(row, 0) = diagonal[row];
    original.upper(row, 0) = upper[row];
  }

  dtdma::prepare_forward_coefficients(original, prepared);
  const dtdma::Scalar forward_lower_row_one = prepared.prepared_lower(1, 0);
  dtdma::prepare_backward_coefficients(original, prepared);

  CHECK(prepared.prepared_lower(0, 0) == 5.0F);
  CHECK(prepared.prepared_lower(1, 0) == forward_lower_row_one);
  CHECK(prepared.prepared_lower(1, 0) == 2.0F);
  CHECK(prepared.prepared_lower(2, 0) == -1.0F);
  CHECK(prepared.prepared_diagonal(0, 0) == Catch::Approx(8.25F));
  CHECK(prepared.prepared_diagonal(1, 0) == 8.0F);
  CHECK(prepared.prepared_diagonal(2, 0) == Catch::Approx(12.5F));
  CHECK(prepared.prepared_upper(0, 0) == Catch::Approx(-1.125F));
  CHECK(prepared.prepared_upper(1, 0) == 3.0F);
  CHECK(prepared.prepared_upper(2, 0) == 7.0F);
}

TEST_CASE("Backward coefficient preparation preserves original and forward state") {
  dtdma::TridiagonalBatch original(5, 2);
  dtdma::PreparedOperatorBatch prepared(5, 2);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    for (std::size_t system = 0; system < original.system_count(); ++system) {
      const auto value = static_cast<dtdma::Scalar>(row * 3 + system + 1);
      original.lower(row, system) = 0.1F * value;
      original.diagonal(row, system) = 10.0F + value;
      original.upper(row, system) = 0.2F * value;
    }
  }

  const std::vector<dtdma::Scalar> original_lower(original.lower().begin(),
                                                   original.lower().end());
  const std::vector<dtdma::Scalar> original_diagonal(
      original.diagonal().begin(), original.diagonal().end());
  const std::vector<dtdma::Scalar> original_upper(original.upper().begin(),
                                                   original.upper().end());

  dtdma::prepare_forward_coefficients(original, prepared);
  const std::vector<dtdma::Scalar> forward_lower(
      prepared.prepared_lower().begin(), prepared.prepared_lower().end());
  const std::vector<dtdma::Scalar> forward_diagonal(
      prepared.prepared_diagonal().begin(),
      prepared.prepared_diagonal().end());
  dtdma::prepare_backward_coefficients(original, prepared);

  CHECK(std::vector<dtdma::Scalar>(original.lower().begin(),
                                   original.lower().end()) == original_lower);
  CHECK(std::vector<dtdma::Scalar>(original.diagonal().begin(),
                                   original.diagonal().end()) ==
        original_diagonal);
  CHECK(std::vector<dtdma::Scalar>(original.upper().begin(),
                                   original.upper().end()) == original_upper);

  for (std::size_t system = 0; system < original.system_count(); ++system) {
    for (std::size_t row = 1; row < original.row_count(); ++row) {
      const std::size_t index =
          dtdma::canonical_index(row, system, original.system_count());
      CHECK(prepared.prepared_diagonal(row, system) ==
            forward_diagonal[index]);
    }

    for (std::size_t row = original.row_count() - 2;
         row < original.row_count(); ++row) {
      const std::size_t index =
          dtdma::canonical_index(row, system, original.system_count());
      CHECK(prepared.prepared_lower(row, system) == forward_lower[index]);
    }
  }
}

TEST_CASE("Backward coefficient preparation rejects invalid dimensions") {
  const dtdma::TridiagonalBatch too_short(2, 1);
  dtdma::PreparedOperatorBatch too_short_prepared(2, 1);
  CHECK_THROWS_AS(
      dtdma::prepare_backward_coefficients(too_short, too_short_prepared),
      std::invalid_argument);

  const dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch wrong_rows(3, 2);
  dtdma::PreparedOperatorBatch wrong_batch_size(4, 3);
  CHECK_THROWS_AS(
      dtdma::prepare_backward_coefficients(original, wrong_rows),
      std::invalid_argument);
  CHECK_THROWS_AS(
      dtdma::prepare_backward_coefficients(original, wrong_batch_size),
      std::invalid_argument);
}

TEST_CASE("The complete prepared operator agrees with the active CaNS reference fixture") {
  // CaNS-World/CaNS main, 245a23348ef795af9aebeda6c767a46ca8be45e8,
  // src/solver_gpu.f90::gaussel_dtdma_gpu_fast_1d, division before multiplication.
  constexpr dtdma::Scalar cans_tolerance = 1.0e-6F;
  const std::array<dtdma::Scalar, 5> original_lower{
      0.75F, -1.25F, 0.875F, -1.5F, 1.125F};
  const std::array<dtdma::Scalar, 5> original_diagonal{
      4.5F, 5.25F, 6.125F, 5.875F, 6.75F};
  const std::array<dtdma::Scalar, 5> original_upper{
      -0.625F, 1.375F, -0.875F, 1.25F, -0.5F};
  const std::array<dtdma::Scalar, 5> cans_prepared_lower{
      0.75F, -1.30050015F, 0.216538385F, 0.0530035309F,
      -0.0105493469F};
  const std::array<dtdma::Scalar, 5> cans_prepared_diagonal{
      4.3451786F, 5.25F, 5.89583349F, 5.65238523F, 6.50121117F};
  const std::array<dtdma::Scalar, 5> cans_prepared_upper{
      -0.0053723529F, -0.0451277643F, 0.193502381F, 1.25F, -0.5F};
  dtdma::TridiagonalBatch original(5, 1);
  dtdma::PreparedOperatorBatch prepared(5, 1);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    original.lower(row, 0) = original_lower[row];
    original.diagonal(row, 0) = original_diagonal[row];
    original.upper(row, 0) = original_upper[row];
  }

  dtdma::prepare_forward_coefficients(original, prepared);
  dtdma::prepare_backward_coefficients(original, prepared);

  for (std::size_t row = 0; row < original.row_count(); ++row) {
    CHECK(prepared.prepared_lower(row, 0) ==
          Catch::Approx(cans_prepared_lower[row]).margin(cans_tolerance));
    CHECK(prepared.prepared_diagonal(row, 0) ==
          Catch::Approx(cans_prepared_diagonal[row]).margin(cans_tolerance));
    CHECK(prepared.prepared_upper(row, 0) ==
          Catch::Approx(cans_prepared_upper[row]).margin(cans_tolerance));
    CHECK(original.lower(row, 0) == original_lower[row]);
    CHECK(original.diagonal(row, 0) == original_diagonal[row]);
    CHECK(original.upper(row, 0) == original_upper[row]);
  }
}
