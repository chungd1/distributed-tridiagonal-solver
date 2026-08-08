#pragma once

#include "dtdma/scalar.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class SharedPreparedOperator {
 public:
  explicit SharedPreparedOperator(const std::size_t row_count)
      : row_count_(checked_row_count(row_count)),
        prepared_lower_(row_count),
        prepared_diagonal_(row_count),
        prepared_upper_(row_count) {}

  [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return row_count_;
  }
  [[nodiscard]] std::size_t storage_system_count() const noexcept {
    return 1;
  }

  [[nodiscard]] Scalar& prepared_lower(const std::size_t row,
                                       const std::size_t) {
    return prepared_lower_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& prepared_lower(
      const std::size_t row,
      const std::size_t) const {
    return prepared_lower_[checked_row(row)];
  }
  [[nodiscard]] Scalar& prepared_diagonal(const std::size_t row,
                                          const std::size_t) {
    return prepared_diagonal_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& prepared_diagonal(
      const std::size_t row,
      const std::size_t) const {
    return prepared_diagonal_[checked_row(row)];
  }
  [[nodiscard]] Scalar& prepared_upper(const std::size_t row,
                                       const std::size_t) {
    return prepared_upper_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& prepared_upper(
      const std::size_t row,
      const std::size_t) const {
    return prepared_upper_[checked_row(row)];
  }

  [[nodiscard]] std::span<Scalar> prepared_lower() noexcept {
    return prepared_lower_;
  }
  [[nodiscard]] std::span<const Scalar> prepared_lower() const noexcept {
    return prepared_lower_;
  }
  [[nodiscard]] std::span<Scalar> prepared_diagonal() noexcept {
    return prepared_diagonal_;
  }
  [[nodiscard]] std::span<const Scalar> prepared_diagonal() const noexcept {
    return prepared_diagonal_;
  }
  [[nodiscard]] std::span<Scalar> prepared_upper() noexcept {
    return prepared_upper_;
  }
  [[nodiscard]] std::span<const Scalar> prepared_upper() const noexcept {
    return prepared_upper_;
  }

 private:
  [[nodiscard]] static std::size_t checked_row_count(
      const std::size_t row_count) {
    if (row_count == 0) {
      throw std::invalid_argument("row count must be greater than zero");
    }
    return row_count;
  }
  [[nodiscard]] std::size_t checked_row(const std::size_t row) const {
    if (row >= row_count_) {
      throw std::out_of_range("row index is out of range");
    }
    return row;
  }
  std::size_t row_count_;
  std::vector<Scalar> prepared_lower_;
  std::vector<Scalar> prepared_diagonal_;
  std::vector<Scalar> prepared_upper_;
};

}  // namespace dtdma
