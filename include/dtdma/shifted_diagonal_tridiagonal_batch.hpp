#pragma once

#include "dtdma/detail/shared_bands_and_rhs.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class ShiftedDiagonalTridiagonalBatch {
 public:
  using PreparedOperator = PreparedOperatorBatch;
  using ReducedOperator = TridiagonalBatch;

  ShiftedDiagonalTridiagonalBatch(const std::size_t row_count,
                                  const std::size_t batch_size)
      : storage_(row_count, batch_size),
        base_diagonal_(row_count),
        shift_(batch_size) {}

  [[nodiscard]] std::size_t row_count() const noexcept {
    return storage_.row_count();
  }
  [[nodiscard]] std::size_t batch_size() const noexcept {
    return storage_.batch_size();
  }
  [[nodiscard]] Scalar& lower(const std::size_t row) {
    return storage_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row) const {
    return storage_.lower(row);
  }
  [[nodiscard]] Scalar& lower(const std::size_t row,
                              const std::size_t system) {
    storage_.check_system(system);
    return storage_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row,
                                    const std::size_t system) const {
    storage_.check_system(system);
    return storage_.lower(row);
  }

  [[nodiscard]] Scalar& base_diagonal(const std::size_t row) {
    return base_diagonal_[checked_row(row)];
  }
  [[nodiscard]] const Scalar& base_diagonal(const std::size_t row) const {
    return base_diagonal_[checked_row(row)];
  }
  [[nodiscard]] Scalar& shift(const std::size_t system) {
    return shift_[checked_system(system)];
  }
  [[nodiscard]] const Scalar& shift(const std::size_t system) const {
    return shift_[checked_system(system)];
  }
  [[nodiscard]] Scalar diagonal(const std::size_t row,
                                const std::size_t system) const {
    return base_diagonal_[checked_row(row)] +
           shift_[checked_system(system)];
  }

  [[nodiscard]] Scalar& upper(const std::size_t row) {
    return storage_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row) const {
    return storage_.upper(row);
  }
  [[nodiscard]] Scalar& upper(const std::size_t row,
                              const std::size_t system) {
    storage_.check_system(system);
    return storage_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row,
                                    const std::size_t system) const {
    storage_.check_system(system);
    return storage_.upper(row);
  }

  [[nodiscard]] Scalar& rhs(const std::size_t row,
                            const std::size_t system) {
    return storage_.rhs(row, system);
  }
  [[nodiscard]] const Scalar& rhs(const std::size_t row,
                                  const std::size_t system) const {
    return storage_.rhs(row, system);
  }

  [[nodiscard]] std::span<Scalar> lower() noexcept {
    return storage_.lower();
  }
  [[nodiscard]] std::span<const Scalar> lower() const noexcept {
    return storage_.lower();
  }
  [[nodiscard]] std::span<Scalar> base_diagonal() noexcept {
    return base_diagonal_;
  }
  [[nodiscard]] std::span<const Scalar> base_diagonal() const noexcept {
    return base_diagonal_;
  }
  [[nodiscard]] std::span<Scalar> shift() noexcept { return shift_; }
  [[nodiscard]] std::span<const Scalar> shift() const noexcept {
    return shift_;
  }
  [[nodiscard]] std::span<Scalar> upper() noexcept {
    return storage_.upper();
  }
  [[nodiscard]] std::span<const Scalar> upper() const noexcept {
    return storage_.upper();
  }
  [[nodiscard]] std::span<Scalar> rhs() noexcept { return storage_.rhs(); }
  [[nodiscard]] std::span<const Scalar> rhs() const noexcept {
    return storage_.rhs();
  }

 private:
  [[nodiscard]] std::size_t checked_row(const std::size_t row) const {
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    return row;
  }
  [[nodiscard]] std::size_t checked_system(
      const std::size_t system) const {
    storage_.check_system(system);
    return system;
  }

  detail::SharedBandsAndRhs storage_;
  std::vector<Scalar> base_diagonal_;
  std::vector<Scalar> shift_;
};

}  // namespace dtdma
