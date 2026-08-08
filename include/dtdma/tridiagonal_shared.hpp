#pragma once

#include "dtdma/detail/shared_bands.hpp"
#include "dtdma/prepared_operator_shared.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class TridiagonalShared {
 public:
  using PreparedOperator = PreparedOperatorShared;
  using ReducedOperator = TridiagonalShared;

  explicit TridiagonalShared(const std::size_t row_count)
      : storage_(row_count), diagonal_(row_count) {}

  [[nodiscard]] std::size_t row_count() const noexcept {
    return storage_.row_count();
  }
  [[nodiscard]] Scalar& lower(const std::size_t row) {
    return storage_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row) const {
    return storage_.lower(row);
  }
  [[nodiscard]] Scalar& lower(const std::size_t row,
                              const std::size_t) {
    return storage_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row,
                                    const std::size_t) const {
    return storage_.lower(row);
  }

  [[nodiscard]] Scalar& diagonal(const std::size_t row) {
    return diagonal_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& diagonal(const std::size_t row) const {
    return diagonal_[checked_row(row)];
  }
  [[nodiscard]] Scalar& diagonal(const std::size_t row,
                                 const std::size_t) {
    return diagonal_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& diagonal(const std::size_t row,
                                       const std::size_t) const {
    return diagonal_[checked_row(row)];
  }

  [[nodiscard]] Scalar& upper(const std::size_t row) {
    return storage_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row) const {
    return storage_.upper(row);
  }
  [[nodiscard]] Scalar& upper(const std::size_t row,
                              const std::size_t) {
    return storage_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row,
                                    const std::size_t) const {
    return storage_.upper(row);
  }

  [[nodiscard]] std::span<Scalar> lower() noexcept {
    return storage_.lower();
  }
  [[nodiscard]] std::span<const Scalar> lower() const noexcept {
    return storage_.lower();
  }
  [[nodiscard]] std::span<Scalar> diagonal() noexcept { return diagonal_; }
  [[nodiscard]] std::span<const Scalar> diagonal() const noexcept {
    return diagonal_;
  }
  [[nodiscard]] std::span<Scalar> upper() noexcept {
    return storage_.upper();
  }
  [[nodiscard]] std::span<const Scalar> upper() const noexcept {
    return storage_.upper();
  }

 private:
  [[nodiscard]] std::size_t checked_row(const std::size_t row) const {
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    return row;
  }

  friend PreparedOperatorShared prepare_operator(
      TridiagonalShared&& original);

  detail::SharedBands storage_;
  std::vector<Scalar> diagonal_;
};

}  // namespace dtdma
