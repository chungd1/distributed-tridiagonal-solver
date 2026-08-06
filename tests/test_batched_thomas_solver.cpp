#include "dtdma/batched_thomas_solver.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace {

constexpr dtdma::Scalar tolerance = 1.0e-4F;

void construct_rhs(dtdma::TridiagonalBatch& batch,
                   const std::vector<dtdma::Scalar>& exact_solution) {
  for (std::size_t row = 0; row < batch.row_count(); ++row) {
    for (std::size_t system = 0; system < batch.batch_size(); ++system) {
      dtdma::Scalar value =
          batch.diagonal(row, system) *
          exact_solution[dtdma::canonical_index(row, system,
                                                 batch.batch_size())];

      if (row > 0) {
        value += batch.lower(row, system) *
                 exact_solution[dtdma::canonical_index(
                     row - 1, system, batch.batch_size())];
      }
      if (row + 1 < batch.row_count()) {
        value += batch.upper(row, system) *
                 exact_solution[dtdma::canonical_index(
                     row + 1, system, batch.batch_size())];
      }

      batch.rhs(row, system) = value;
    }
  }
}

void check_solution(const dtdma::TridiagonalBatch& batch,
                    const std::vector<dtdma::Scalar>& exact_solution) {
  for (std::size_t row = 0; row < batch.row_count(); ++row) {
    for (std::size_t system = 0; system < batch.batch_size(); ++system) {
      const std::size_t index =
          dtdma::canonical_index(row, system, batch.batch_size());
      CHECK(batch.rhs(row, system) ==
            Catch::Approx(exact_solution[index]).margin(tolerance));
    }
  }
}

}  // namespace

TEST_CASE("The batched Thomas solver solves constant-coefficient systems") {
  constexpr std::size_t row_count = 6;

  for (const std::size_t batch_size : std::array<std::size_t, 3>{1, 2, 5}) {
    dtdma::TridiagonalBatch batch(row_count, batch_size);
    std::vector<dtdma::Scalar> exact_solution(batch.element_count());

    for (std::size_t row = 0; row < row_count; ++row) {
      for (std::size_t system = 0; system < batch_size; ++system) {
        batch.lower(row, system) = row == 0 ? 0.0F : -1.0F;
        batch.diagonal(row, system) = 4.0F;
        batch.upper(row, system) = row + 1 == row_count ? 0.0F : -1.0F;
        exact_solution[dtdma::canonical_index(row, system, batch_size)] =
            static_cast<dtdma::Scalar>(10 * (system + 1) + row);
      }
    }

    construct_rhs(batch, exact_solution);
    dtdma::batched_thomas_solve(batch);
    check_solution(batch, exact_solution);
  }
}

TEST_CASE("The batched Thomas solver handles row-varying coefficients") {
  constexpr std::size_t row_count = 7;
  constexpr std::size_t batch_size = 3;
  dtdma::TridiagonalBatch batch(row_count, batch_size);
  std::vector<dtdma::Scalar> exact_solution(batch.element_count());

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const auto row_value = static_cast<dtdma::Scalar>(row);
      batch.lower(row, system) = row == 0 ? 0.0F : -0.15F * row_value;
      batch.diagonal(row, system) = 3.0F + 0.4F * row_value;
      batch.upper(row, system) =
          row + 1 == row_count ? 0.0F : 0.1F * (row_value + 1.0F);
      exact_solution[dtdma::canonical_index(row, system, batch_size)] =
          static_cast<dtdma::Scalar>((system + 1) * (row + 2)) - 4.0F;
    }
  }

  construct_rhs(batch, exact_solution);
  dtdma::batched_thomas_solve(batch);
  check_solution(batch, exact_solution);
}

TEST_CASE("The batched Thomas solver handles system-varying coefficients") {
  constexpr std::size_t row_count = 5;
  constexpr std::size_t batch_size = 4;
  dtdma::TridiagonalBatch batch(row_count, batch_size);
  std::vector<dtdma::Scalar> exact_solution(batch.element_count());

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const auto system_value = static_cast<dtdma::Scalar>(system);
      batch.lower(row, system) =
          row == 0 ? 0.0F : -0.2F * (system_value + 1.0F);
      batch.diagonal(row, system) = 3.5F + system_value;
      batch.upper(row, system) =
          row + 1 == row_count ? 0.0F : 0.1F * (system_value + 1.0F);
      exact_solution[dtdma::canonical_index(row, system, batch_size)] =
          static_cast<dtdma::Scalar>(20 * system + 3 * row + 1);
    }
  }

  construct_rhs(batch, exact_solution);
  dtdma::batched_thomas_solve(batch);
  check_solution(batch, exact_solution);
}

TEST_CASE("The batched Thomas solver solves deterministic random systems") {
  constexpr std::size_t row_count = 9;
  constexpr std::size_t batch_size = 6;
  dtdma::TridiagonalBatch batch(row_count, batch_size);
  std::vector<dtdma::Scalar> exact_solution(batch.element_count());
  std::mt19937 generator(1729U);
  std::uniform_real_distribution<dtdma::Scalar> coefficient_distribution(
      -0.75F, 0.75F);
  std::uniform_real_distribution<dtdma::Scalar> solution_distribution(-3.0F,
                                                                       3.0F);

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const dtdma::Scalar lower =
          row == 0 ? 0.0F : coefficient_distribution(generator);
      const dtdma::Scalar upper = row + 1 == row_count
                                      ? 0.0F
                                      : coefficient_distribution(generator);
      batch.lower(row, system) = lower;
      batch.upper(row, system) = upper;
      batch.diagonal(row, system) =
          std::abs(lower) + std::abs(upper) + 1.0F;
      exact_solution[dtdma::canonical_index(row, system, batch_size)] =
          solution_distribution(generator);
    }
  }

  construct_rhs(batch, exact_solution);
  dtdma::batched_thomas_solve(batch);
  check_solution(batch, exact_solution);
}

TEST_CASE("The batched Thomas solver preserves coefficients and overwrites RHS") {
  constexpr std::size_t row_count = 4;
  constexpr std::size_t batch_size = 3;
  dtdma::TridiagonalBatch batch(row_count, batch_size);
  std::vector<dtdma::Scalar> exact_solution(batch.element_count());

  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      batch.lower(row, system) = row == 0 ? 0.0F : -1.0F;
      batch.diagonal(row, system) = 4.0F;
      batch.upper(row, system) = row + 1 == row_count ? 0.0F : -1.0F;
      exact_solution[dtdma::canonical_index(row, system, batch_size)] =
          static_cast<dtdma::Scalar>(7 * system + 2 * row + 1);
    }
  }

  construct_rhs(batch, exact_solution);
  const std::vector<dtdma::Scalar> original_lower(batch.lower().begin(),
                                                  batch.lower().end());
  const std::vector<dtdma::Scalar> original_diagonal(batch.diagonal().begin(),
                                                     batch.diagonal().end());
  const std::vector<dtdma::Scalar> original_upper(batch.upper().begin(),
                                                  batch.upper().end());
  const std::vector<dtdma::Scalar> original_rhs(batch.rhs().begin(),
                                                batch.rhs().end());

  dtdma::batched_thomas_solve(batch);

  CHECK(std::vector<dtdma::Scalar>(batch.lower().begin(), batch.lower().end()) ==
        original_lower);
  CHECK(std::vector<dtdma::Scalar>(batch.diagonal().begin(),
                                   batch.diagonal().end()) == original_diagonal);
  CHECK(std::vector<dtdma::Scalar>(batch.upper().begin(), batch.upper().end()) ==
        original_upper);
  CHECK(std::vector<dtdma::Scalar>(batch.rhs().begin(), batch.rhs().end()) !=
        original_rhs);
  check_solution(batch, exact_solution);
}
