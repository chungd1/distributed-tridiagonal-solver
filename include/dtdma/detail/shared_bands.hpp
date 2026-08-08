#pragma once

#include "dtdma/scalar.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma::detail {

class SharedBands {
 public:
  SharedBands() = default;

  explicit SharedBands(const std::size_t row_count)
      : row_count_(checked_row_count(row_count)),
        lower_(row_count),
        upper_(row_count) {}

  [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

  [[nodiscard]] Scalar& lower(const std::size_t row) {
    return lower_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row) const {
    return lower_[checked_row(row)];
  }
  [[nodiscard]] Scalar& upper(const std::size_t row) {
    return upper_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row) const {
    return upper_[checked_row(row)];
  }

  [[nodiscard]] std::span<Scalar> lower() noexcept { return lower_; }
  [[nodiscard]] std::span<const Scalar> lower() const noexcept {
    return lower_;
  }
  [[nodiscard]] std::span<Scalar> upper() noexcept { return upper_; }
  [[nodiscard]] std::span<const Scalar> upper() const noexcept {
    return upper_;
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
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    return row;
  }

  std::size_t row_count_{};
  std::vector<Scalar> lower_;
  std::vector<Scalar> upper_;
};

}  // namespace dtdma::detail
