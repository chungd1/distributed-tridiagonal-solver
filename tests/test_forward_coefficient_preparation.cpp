#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
  CHECK(prepared.batch_size() == original.batch_size());
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
    for (std::size_t system = 0; system < original.batch_size(); ++system) {
      const auto value = static_cast<dtdma::Scalar>(row * 3 + system + 1);
      original.lower(row, system) = value;
      original.diagonal(row, system) = value + 10.0F;
      original.upper(row, system) = value + 2.0F;
      original.rhs(row, system) = value + 20.0F;
    }
  }

  const std::vector<dtdma::Scalar> lower(original.lower().begin(),
                                          original.lower().end());
  const std::vector<dtdma::Scalar> diagonal(original.diagonal().begin(),
                                             original.diagonal().end());
  const std::vector<dtdma::Scalar> upper(original.upper().begin(),
                                          original.upper().end());
  const std::vector<dtdma::Scalar> rhs(original.rhs().begin(),
                                        original.rhs().end());

  dtdma::prepare_forward_coefficients(original, prepared);

  CHECK(std::vector<dtdma::Scalar>(original.lower().begin(),
                                   original.lower().end()) == lower);
  CHECK(std::vector<dtdma::Scalar>(original.diagonal().begin(),
                                   original.diagonal().end()) == diagonal);
  CHECK(std::vector<dtdma::Scalar>(original.upper().begin(),
                                   original.upper().end()) == upper);
  CHECK(std::vector<dtdma::Scalar>(original.rhs().begin(),
                                   original.rhs().end()) == rhs);
}

TEST_CASE("Forward coefficient preparation leaves other entries initialized") {
  dtdma::TridiagonalBatch original(4, 2);
  dtdma::PreparedOperatorBatch prepared(4, 2);

  for (std::size_t system = 0; system < original.batch_size(); ++system) {
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

  for (std::size_t system = 0; system < original.batch_size(); ++system) {
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
