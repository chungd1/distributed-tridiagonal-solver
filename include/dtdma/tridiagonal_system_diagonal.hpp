#pragma once

#include "dtdma/detail/shared_bands.hpp"
#include "dtdma/indexing.hpp"
#include "dtdma/prepared_operator_batch_shared_offdiagonals.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class TridiagonalSystemDiagonal {
 public:
  using PreparedOperator = PreparedOperatorBatchSharedOffdiagonals;
  using ReducedOperator = TridiagonalBatch;

  TridiagonalSystemDiagonal(const std::size_t row_count,
                            const std::size_t system_count)
      : storage_(row_count),
        system_count_(checked_system_count(system_count)),
        diagonal_(checked_element_count(row_count, system_count)) {}

  [[nodiscard]] std::size_t row_count() const noexcept {
    return storage_.row_count();
  }
  [[nodiscard]] std::size_t system_count() const noexcept {
    return system_count_;
  }
  [[nodiscard]] Scalar& lower(const std::size_t row) {
    return storage_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row) const {
    return storage_.lower(row);
  }
  [[nodiscard]] Scalar& lower(const std::size_t row,
                              const std::size_t system) {
    check_system(system);
    return storage_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row,
                                    const std::size_t system) const {
    check_system(system);
    return storage_.lower(row);
  }

  [[nodiscard]] Scalar& diagonal(const std::size_t row,
                                 const std::size_t system) {
    return diagonal_[checked_index(row, system)];
  }
  [[nodiscard]] const Scalar& diagonal(const std::size_t row,
                                       const std::size_t system) const {
    return diagonal_[checked_index(row, system)];
  }

  [[nodiscard]] Scalar& upper(const std::size_t row) {
    return storage_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row) const {
    return storage_.upper(row);
  }
  [[nodiscard]] Scalar& upper(const std::size_t row,
                              const std::size_t system) {
    check_system(system);
    return storage_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row,
                                    const std::size_t system) const {
    check_system(system);
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
  [[nodiscard]] static std::size_t checked_system_count(
      const std::size_t system_count) {
    if (system_count == 0) {
      throw std::invalid_argument("system count must be greater than zero");
    }
    return system_count;
  }
  [[nodiscard]] static std::size_t checked_element_count(
      const std::size_t row_count,
      const std::size_t system_count) {
    if (system_count == 0) {
      throw std::invalid_argument("system count must be greater than zero");
    }
    if (row_count > std::numeric_limits<std::size_t>::max() / system_count) {
      throw std::length_error(
          "system-diagonal coefficient element count overflows");
    }
    return row_count * system_count;
  }

  [[nodiscard]] std::size_t checked_index(const std::size_t row,
                                          const std::size_t system) const {
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    check_system(system);
    return canonical_index(row, system, system_count());
  }
  void check_system(const std::size_t system) const {
    if (system >= system_count_) {
      throw std::out_of_range("system index is out of range");
    }
  }

  friend PreparedOperatorBatchSharedOffdiagonals prepare_operator(
      TridiagonalSystemDiagonal&& original);

  detail::SharedBands storage_;
  std::size_t system_count_;
  std::vector<Scalar> diagonal_;
};

}  // namespace dtdma
