#pragma once

#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma::detail {

class SharedBandsAndRhs {
 public:
  SharedBandsAndRhs(const std::size_t row_count,
                    const std::size_t batch_size)
      : rhs_(row_count, batch_size),
        lower_(row_count),
        upper_(row_count) {}

  [[nodiscard]] std::size_t row_count() const noexcept {
    return rhs_.row_count();
  }
  [[nodiscard]] std::size_t batch_size() const noexcept {
    return rhs_.batch_size();
  }
  [[nodiscard]] std::size_t rhs_element_count() const noexcept {
    return rhs_.element_count();
  }

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

  [[nodiscard]] Scalar& rhs(const std::size_t row,
                            const std::size_t system) {
    return rhs_.rhs(row, system);
  }
  [[nodiscard]] const Scalar& rhs(const std::size_t row,
                                  const std::size_t system) const {
    return rhs_.rhs(row, system);
  }

  void check_system(const std::size_t system) const {
    if (system >= batch_size()) {
      throw std::out_of_range("system index is out of range");
    }
  }

  [[nodiscard]] std::span<Scalar> lower() noexcept { return lower_; }
  [[nodiscard]] std::span<const Scalar> lower() const noexcept {
    return lower_;
  }
  [[nodiscard]] std::span<Scalar> upper() noexcept { return upper_; }
  [[nodiscard]] std::span<const Scalar> upper() const noexcept {
    return upper_;
  }
  [[nodiscard]] std::span<Scalar> rhs() noexcept { return rhs_.rhs(); }
  [[nodiscard]] std::span<const Scalar> rhs() const noexcept {
    return rhs_.rhs();
  }

 private:
  [[nodiscard]] std::size_t checked_row(const std::size_t row) const {
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    return row;
  }

  ReducedRhsBatch rhs_;
  std::vector<Scalar> lower_;
  std::vector<Scalar> upper_;
};

}  // namespace dtdma::detail
