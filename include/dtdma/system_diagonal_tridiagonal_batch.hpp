#pragma once

#include "dtdma/detail/shared_bands_and_rhs.hpp"
#include "dtdma/indexing.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/scalar.hpp"
#include "dtdma/tridiagonal_batch.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class SystemDiagonalTridiagonalBatch {
 public:
  using PreparedOperator = PreparedOperatorBatch;
  using ReducedOperator = TridiagonalBatch;

  SystemDiagonalTridiagonalBatch(const std::size_t row_count,
                                 const std::size_t batch_size)
      : storage_(row_count, batch_size),
        diagonal_(storage_.rhs_element_count()) {}

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
  [[nodiscard]] std::span<Scalar> rhs() noexcept { return storage_.rhs(); }
  [[nodiscard]] std::span<const Scalar> rhs() const noexcept {
    return storage_.rhs();
  }

 private:
  [[nodiscard]] std::size_t checked_index(const std::size_t row,
                                          const std::size_t system) const {
    if (row >= row_count()) {
      throw std::out_of_range("row index is out of range");
    }
    storage_.check_system(system);
    return canonical_index(row, system, batch_size());
  }

  detail::SharedBandsAndRhs storage_;
  std::vector<Scalar> diagonal_;
};

}  // namespace dtdma
