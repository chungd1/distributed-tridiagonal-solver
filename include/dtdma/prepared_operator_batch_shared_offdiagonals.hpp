#pragma once

#include "dtdma/detail/shared_bands.hpp"
#include "dtdma/indexing.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class TridiagonalShiftedDiagonal;
class TridiagonalSystemDiagonal;
class PreparedOperatorBatchSharedOffdiagonals;

PreparedOperatorBatchSharedOffdiagonals prepare_operator(
    TridiagonalShiftedDiagonal&& original);
PreparedOperatorBatchSharedOffdiagonals prepare_operator(
    TridiagonalSystemDiagonal&& original);

class PreparedOperatorBatchSharedOffdiagonals {
 public:
  PreparedOperatorBatchSharedOffdiagonals(
      const std::size_t row_count,
      const std::size_t system_count)
      : row_count_(row_count),
        system_count_(system_count),
        element_count_(checked_element_count(row_count, system_count)),
        prepared_lower_(element_count_),
        prepared_diagonal_(element_count_),
        prepared_upper_(element_count_) {}

  [[nodiscard]] std::size_t row_count() const noexcept {
    return row_count_;
  }
  [[nodiscard]] std::size_t system_count() const noexcept {
    return system_count_;
  }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return element_count_;
  }
  [[nodiscard]] std::size_t storage_system_count() const noexcept {
    return system_count_;
  }

  [[nodiscard]] Scalar& lower(const std::size_t row,
                              const std::size_t system) {
    check_system(system);
    return retained_bands_.lower(row);
  }
  [[nodiscard]] const Scalar& lower(const std::size_t row,
                                    const std::size_t system) const {
    check_system(system);
    return retained_bands_.lower(row);
  }
  [[nodiscard]] Scalar& upper(const std::size_t row,
                              const std::size_t system) {
    check_system(system);
    return retained_bands_.upper(row);
  }
  [[nodiscard]] const Scalar& upper(const std::size_t row,
                                    const std::size_t system) const {
    check_system(system);
    return retained_bands_.upper(row);
  }

  [[nodiscard]] Scalar& prepared_lower(const std::size_t row,
                                       const std::size_t system) {
    return prepared_lower_[checked_index(row, system)];
  }
  [[nodiscard]] const Scalar& prepared_lower(
      const std::size_t row,
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
  [[nodiscard]] const Scalar& prepared_upper(
      const std::size_t row,
      const std::size_t system) const {
    return prepared_upper_[checked_index(row, system)];
  }

  [[nodiscard]] std::span<Scalar> lower() noexcept {
    return retained_bands_.lower();
  }
  [[nodiscard]] std::span<const Scalar> lower() const noexcept {
    return retained_bands_.lower();
  }
  [[nodiscard]] std::span<Scalar> upper() noexcept {
    return retained_bands_.upper();
  }
  [[nodiscard]] std::span<const Scalar> upper() const noexcept {
    return retained_bands_.upper();
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
      const std::size_t system_count) {
    if (row_count == 0) {
      throw std::invalid_argument("row count must be greater than zero");
    }
    if (system_count == 0) {
      throw std::invalid_argument("system count must be greater than zero");
    }
    if (row_count > std::numeric_limits<std::size_t>::max() / system_count) {
      throw std::length_error(
          "prepared operator batch element count overflows");
    }
    return row_count * system_count;
  }

  [[nodiscard]] std::size_t checked_index(const std::size_t row,
                                          const std::size_t system) const {
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    check_system(system);
    return canonical_index(row, system, system_count_);
  }
  void check_system(const std::size_t system) const {
    if (system >= system_count_) {
      throw std::out_of_range("system index is out of range");
    }
  }

  friend PreparedOperatorBatchSharedOffdiagonals prepare_operator(
      TridiagonalShiftedDiagonal&& original);
  friend PreparedOperatorBatchSharedOffdiagonals prepare_operator(
      TridiagonalSystemDiagonal&& original);

  detail::SharedBands retained_bands_;
  std::size_t row_count_;
  std::size_t system_count_;
  std::size_t element_count_;
  std::vector<Scalar> prepared_lower_;
  std::vector<Scalar> prepared_diagonal_;
  std::vector<Scalar> prepared_upper_;
};

}  // namespace dtdma
