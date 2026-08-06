#pragma once

#include "dtdma/indexing.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class ReducedRhsBatch {
 public:
  ReducedRhsBatch(const std::size_t row_count,
                  const std::size_t batch_size)
      : row_count_(row_count),
        batch_size_(batch_size),
        element_count_(checked_element_count(row_count, batch_size)),
        rhs_(element_count_) {}

  [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
  [[nodiscard]] std::size_t batch_size() const noexcept { return batch_size_; }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return element_count_;
  }

  [[nodiscard]] Scalar& rhs(const std::size_t row,
                            const std::size_t system) {
    return rhs_[checked_index(row, system)];
  }

  [[nodiscard]] const Scalar& rhs(const std::size_t row,
                                  const std::size_t system) const {
    return rhs_[checked_index(row, system)];
  }

  [[nodiscard]] std::span<Scalar> rhs() noexcept { return rhs_; }
  [[nodiscard]] std::span<const Scalar> rhs() const noexcept { return rhs_; }

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
      throw std::length_error("reduced RHS batch element count overflows");
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
  std::vector<Scalar> rhs_;
};

}  // namespace dtdma
