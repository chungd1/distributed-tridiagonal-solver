#pragma once

#include "dtdma/indexing.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class PreparedOperatorBatch {
 public:
  PreparedOperatorBatch(const std::size_t row_count,
                        const std::size_t batch_size)
      : row_count_(row_count),
        batch_size_(batch_size),
        element_count_(checked_element_count(row_count, batch_size)),
        prepared_lower_(element_count_),
        prepared_diagonal_(element_count_),
        prepared_upper_(element_count_) {}

  [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
  [[nodiscard]] std::size_t batch_size() const noexcept { return batch_size_; }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return element_count_;
  }

  [[nodiscard]] Scalar& prepared_lower(const std::size_t row,
                                       const std::size_t system) {
    return prepared_lower_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& prepared_lower(const std::size_t row,
                                             const std::size_t system) const {
    return prepared_lower_[checked_index(row, system)];
  }

  [[nodiscard]] Scalar& prepared_diagonal(const std::size_t row,
                                          const std::size_t system) {
    return prepared_diagonal_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& prepared_diagonal(
      const std::size_t row,
      const std::size_t system) const {
    return prepared_diagonal_[checked_index(row, system)];
  }

  [[nodiscard]] Scalar& prepared_upper(const std::size_t row,
                                       const std::size_t system) {
    return prepared_upper_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& prepared_upper(const std::size_t row,
                                             const std::size_t system) const {
    return prepared_upper_[checked_index(row, system)];
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
  [[nodiscard]] static std::size_t checked_element_count(
      const std::size_t row_count,
      const std::size_t batch_size) {
    if (row_count == 0) {
      throw std::invalid_argument("row count must be greater than zero");
    }
    if (batch_size == 0) {
      throw std::invalid_argument("batch size must be greater than zero");
    }
    if (row_count > std::numeric_limits<std::size_t>::max() / batch_size) {
      throw std::length_error("prepared operator batch element count overflows");
    }
    return row_count * batch_size;
  }

  [[nodiscard]] std::size_t checked_index(const std::size_t row,
                                          const std::size_t system) const {
    if (row >= row_count_) {
      throw std::out_of_range("row index is out of range");
    }
    if (system >= batch_size_) {
      throw std::out_of_range("system index is out of range");
    }
    return canonical_index(row, system, batch_size_);
  }

  std::size_t row_count_;
  std::size_t batch_size_;
  std::size_t element_count_;
  std::vector<Scalar> prepared_lower_;
  std::vector<Scalar> prepared_diagonal_;
  std::vector<Scalar> prepared_upper_;
};

}  // namespace dtdma
