#include "dtdma/indexing.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

TEST_CASE("A tridiagonal batch has the requested dimensions") {
  const dtdma::TridiagonalBatch batch(4, 3);

  CHECK(batch.row_count() == 4);
  CHECK(batch.batch_size() == 3);
  CHECK(batch.element_count() == batch.row_count() * batch.batch_size());
  CHECK(batch.lower().size() == batch.element_count());
  CHECK(batch.diagonal().size() == batch.element_count());
  CHECK(batch.upper().size() == batch.element_count());
  CHECK(batch.rhs().size() == batch.element_count());
}

TEST_CASE("A tridiagonal batch rejects invalid dimensions") {
  CHECK_THROWS_AS(dtdma::TridiagonalBatch(0, 3), std::invalid_argument);
  CHECK_THROWS_AS(dtdma::TridiagonalBatch(4, 0), std::invalid_argument);

  const auto maximum = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(dtdma::TridiagonalBatch(maximum, 2), std::length_error);
}

TEST_CASE("Canonical indexing keeps systems contiguous within a row") {
  dtdma::TridiagonalBatch batch(3, 4);

  REQUIRE(dtdma::canonical_index(2, 1, batch.batch_size()) == 9);

  auto* first_system = &batch.diagonal(1, 0);
  auto* adjacent_system = &batch.diagonal(1, 1);
  CHECK(adjacent_system == first_system + 1);

  batch.diagonal(2, 1) = 7.5F;
  CHECK(batch.diagonal()[9] == 7.5F);
}

TEST_CASE("All coefficient and right-hand-side arrays are mutable") {
  dtdma::TridiagonalBatch batch(2, 2);

  STATIC_CHECK(std::is_same_v<dtdma::Scalar, float>);
  STATIC_CHECK(
      std::is_same_v<decltype(batch.lower()), std::span<dtdma::Scalar>>);
  STATIC_CHECK(
      std::is_same_v<decltype(batch.diagonal()), std::span<dtdma::Scalar>>);
  STATIC_CHECK(
      std::is_same_v<decltype(batch.upper()), std::span<dtdma::Scalar>>);
  STATIC_CHECK(
      std::is_same_v<decltype(batch.rhs()), std::span<dtdma::Scalar>>);

  batch.lower(1, 0) = 1.0F;
  batch.diagonal(1, 0) = 2.0F;
  batch.upper(1, 0) = 3.0F;
  batch.rhs(1, 0) = 4.0F;

  const auto index = dtdma::canonical_index(1, 0, batch.batch_size());
  CHECK(batch.lower()[index] == 1.0F);
  CHECK(batch.diagonal()[index] == 2.0F);
  CHECK(batch.upper()[index] == 3.0F);
  CHECK(batch.rhs()[index] == 4.0F);
}

TEST_CASE("All arrays provide const scalar and contiguous access") {
  dtdma::TridiagonalBatch mutable_batch(1, 1);
  mutable_batch.lower(0, 0) = 1.0F;
  mutable_batch.diagonal(0, 0) = 2.0F;
  mutable_batch.upper(0, 0) = 3.0F;
  mutable_batch.rhs(0, 0) = 4.0F;
  const dtdma::TridiagonalBatch& batch = mutable_batch;

  STATIC_CHECK(std::is_same_v<decltype(batch.lower()),
                              std::span<const dtdma::Scalar>>);
  STATIC_CHECK(std::is_same_v<decltype(batch.diagonal()),
                              std::span<const dtdma::Scalar>>);
  STATIC_CHECK(std::is_same_v<decltype(batch.upper()),
                              std::span<const dtdma::Scalar>>);
  STATIC_CHECK(std::is_same_v<decltype(batch.rhs()),
                              std::span<const dtdma::Scalar>>);
  CHECK(batch.lower(0, 0) == 1.0F);
  CHECK(batch.diagonal(0, 0) == 2.0F);
  CHECK(batch.upper(0, 0) == 3.0F);
  CHECK(batch.rhs(0, 0) == 4.0F);
}

TEST_CASE("Storage is value-initialized") {
  const dtdma::TridiagonalBatch batch(3, 2);

  for (const auto value : batch.lower()) {
    CHECK(value == 0.0F);
  }
  for (const auto value : batch.diagonal()) {
    CHECK(value == 0.0F);
  }
  for (const auto value : batch.upper()) {
    CHECK(value == 0.0F);
  }
  for (const auto value : batch.rhs()) {
    CHECK(value == 0.0F);
  }
}

TEST_CASE("Indexed access checks row and system bounds") {
  dtdma::TridiagonalBatch batch(2, 3);
  const dtdma::TridiagonalBatch& const_batch = batch;

  CHECK_THROWS_AS(batch.lower(2, 0), std::out_of_range);
  CHECK_THROWS_AS(batch.diagonal(0, 3), std::out_of_range);
  CHECK_THROWS_AS(const_batch.upper(2, 0), std::out_of_range);
  CHECK_THROWS_AS(const_batch.rhs(0, 3), std::out_of_range);
}
