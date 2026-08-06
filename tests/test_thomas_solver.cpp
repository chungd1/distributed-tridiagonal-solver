#include "dtdma/scalar.hpp"
#include "dtdma/thomas_solver.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

TEST_CASE("The Thomas solver solves a two-row tridiagonal system exactly") {
  const std::array<dtdma::Scalar, 2> lower{0.0F, 1.0F};
  std::array<dtdma::Scalar, 2> diagonal{2.0F, 3.0F};
  const std::array<dtdma::Scalar, 2> upper{1.0F, 0.0F};
  std::array<dtdma::Scalar, 2> rhs{4.0F, 7.0F};

  dtdma::thomas_solve(lower, diagonal, upper, rhs);

  CHECK(rhs[0] == 1.0F);
  CHECK(rhs[1] == 2.0F);
}

TEST_CASE("The Thomas solver solves a manufactured five-row system") {
  const std::array<dtdma::Scalar, 5> lower{0.0F, -1.0F, -1.0F, -1.0F,
                                           -1.0F};
  std::array<dtdma::Scalar, 5> diagonal{2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
  const std::array<dtdma::Scalar, 5> upper{-1.0F, -1.0F, -1.0F, -1.0F,
                                           0.0F};
  std::array<dtdma::Scalar, 5> rhs{0.0F, 0.0F, 0.0F, 0.0F, 6.0F};
  const std::array<dtdma::Scalar, 5> expected{1.0F, 2.0F, 3.0F, 4.0F,
                                              5.0F};

  dtdma::thomas_solve(lower, diagonal, upper, rhs);

  for (std::size_t row = 0; row < rhs.size(); ++row) {
    CHECK(rhs[row] == Catch::Approx(expected[row]));
  }
}

TEST_CASE("The Thomas solver modifies only its working arrays") {
  const std::array<dtdma::Scalar, 3> lower{0.0F, -1.0F, -1.0F};
  std::array<dtdma::Scalar, 3> diagonal{2.0F, 2.0F, 2.0F};
  const std::array<dtdma::Scalar, 3> upper{-1.0F, -1.0F, 0.0F};
  std::array<dtdma::Scalar, 3> rhs{0.0F, 0.0F, 4.0F};
  const auto original_lower = lower;
  const auto original_diagonal = diagonal;
  const auto original_upper = upper;
  const auto original_rhs = rhs;

  dtdma::thomas_solve(lower, diagonal, upper, rhs);

  CHECK(lower == original_lower);
  CHECK(upper == original_upper);
  CHECK(diagonal != original_diagonal);
  CHECK(rhs != original_rhs);
  CHECK(rhs[0] == Catch::Approx(1.0F));
  CHECK(rhs[1] == Catch::Approx(2.0F));
  CHECK(rhs[2] == Catch::Approx(3.0F));
}

TEST_CASE("The Thomas solver rejects mismatched array lengths") {
  const std::array<dtdma::Scalar, 3> coefficients{0.0F, 1.0F, 0.0F};
  std::array<dtdma::Scalar, 3> diagonal{2.0F, 2.0F, 2.0F};
  std::array<dtdma::Scalar, 3> rhs{1.0F, 1.0F, 1.0F};

  CHECK_THROWS_AS(
      dtdma::thomas_solve(std::span<const dtdma::Scalar>{coefficients}.first(2),
                          diagonal, coefficients, rhs),
      std::invalid_argument);
  CHECK_THROWS_AS(
      dtdma::thomas_solve(coefficients,
                          std::span<dtdma::Scalar>{diagonal}.first(2),
                          coefficients, rhs),
      std::invalid_argument);
  CHECK_THROWS_AS(
      dtdma::thomas_solve(coefficients, diagonal,
                          std::span<const dtdma::Scalar>{coefficients}.first(2),
                          rhs),
      std::invalid_argument);
  CHECK_THROWS_AS(
      dtdma::thomas_solve(coefficients, diagonal, coefficients,
                          std::span<dtdma::Scalar>{rhs}.first(2)),
      std::invalid_argument);
}

TEST_CASE("The Thomas solver rejects systems with fewer than two rows") {
  const std::array<dtdma::Scalar, 1> lower{0.0F};
  std::array<dtdma::Scalar, 1> diagonal{1.0F};
  const std::array<dtdma::Scalar, 1> upper{0.0F};
  std::array<dtdma::Scalar, 1> rhs{1.0F};

  CHECK_THROWS_AS(dtdma::thomas_solve(lower, diagonal, upper, rhs),
                  std::invalid_argument);

  CHECK_THROWS_AS(
      dtdma::thomas_solve(std::span<const dtdma::Scalar>{},
                          std::span<dtdma::Scalar>{},
                          std::span<const dtdma::Scalar>{},
                          std::span<dtdma::Scalar>{}),
      std::invalid_argument);
}
