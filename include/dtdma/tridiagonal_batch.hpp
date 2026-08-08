#pragma once

#include "dtdma/indexing.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class PreparedOperatorBatch;
class TridiagonalBatch;

PreparedOperatorBatch prepare_operator(TridiagonalBatch&& original);

class TridiagonalBatch {
 public:
  using PreparedOperator = PreparedOperatorBatch;
  using ReducedOperator = TridiagonalBatch;

  TridiagonalBatch(const std::size_t row_count,
                   const std::size_t system_count)
      : row_count_(row_count),
        system_count_(system_count),
        element_count_(checked_element_count(row_count, system_count)),
        lower_(element_count_),
        diagonal_(element_count_),
        upper_(element_count_) {}

  [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
  [[nodiscard]] std::size_t system_count() const noexcept {
    return system_count_;
  }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return element_count_;
  }

  [[nodiscard]] Scalar& lower(const std::size_t row,
                              const std::size_t system) {
    return lower_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& lower(const std::size_t row,
                                    const std::size_t system) const {
    return lower_[checked_index(row, system)];
  }

  [[nodiscard]] Scalar& diagonal(const std::size_t row,
                                 const std::size_t system) {
    return diagonal_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& diagonal(const std::size_t row,
                                       const std::size_t system) const {
    return diagonal_[checked_index(row, system)];
  }

  [[nodiscard]] Scalar& upper(const std::size_t row,
                              const std::size_t system) {
    return upper_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& upper(const std::size_t row,
                                    const std::size_t system) const {
    return upper_[checked_index(row, system)];
  }

  [[nodiscard]] std::span<Scalar> lower() noexcept { return lower_; }
  [[nodiscard]] std::span<const Scalar> lower() const noexcept { return lower_; }

  [[nodiscard]] std::span<Scalar> diagonal() noexcept { return diagonal_; }
  [[nodiscard]] std::span<const Scalar> diagonal() const noexcept {
    return diagonal_;
  }

  [[nodiscard]] std::span<Scalar> upper() noexcept { return upper_; }
  [[nodiscard]] std::span<const Scalar> upper() const noexcept { return upper_; }

 private:
  [[nodiscard]] static std::size_t checked_element_count(
      const std::size_t row_count,
      const std::size_t system_count) {
    if (row_count == 0) {
      throw std::invalid_argument("row count must be greater than zero");
    }
    if (system_count == 0) {
      throw std::invalid_argument("system count must be greater than zero");
    }
    if (row_count > std::numeric_limits<std::size_t>::max() / system_count) {
      throw std::length_error("tridiagonal batch element count overflows");
    }
    return row_count * system_count;
  }

  [[nodiscard]] std::size_t checked_index(const std::size_t row,
                                          const std::size_t system) const {
    if (row >= row_count_) {
      throw std::out_of_range("row index is out of range");
    }
    if (system >= system_count_) {
      throw std::out_of_range("system index is out of range");
    }
    return canonical_index(row, system, system_count_);
  }

  friend PreparedOperatorBatch prepare_operator(
      TridiagonalBatch&& original);

  std::size_t row_count_;
  std::size_t system_count_;
  std::size_t element_count_;
  std::vector<Scalar> lower_;
  std::vector<Scalar> diagonal_;
  std::vector<Scalar> upper_;
};

}  // namespace dtdma
