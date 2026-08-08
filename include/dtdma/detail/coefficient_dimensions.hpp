#pragma once

#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/shared_prepared_operator.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/tridiagonal_shared.hpp"
#include "dtdma/tridiagonal_shifted_diagonal.hpp"
#include "dtdma/tridiagonal_system_diagonal.hpp"

#include <cstddef>

namespace dtdma::detail {

inline std::size_t storage_system_count(
    const TridiagonalShared&) noexcept {
  return 1;
}

inline std::size_t storage_system_count(
    const TridiagonalShiftedDiagonal& coefficients) noexcept {
  return coefficients.system_count();
}

inline std::size_t storage_system_count(
    const TridiagonalSystemDiagonal& coefficients) noexcept {
  return coefficients.system_count();
}

inline std::size_t storage_system_count(
    const TridiagonalBatch& coefficients) noexcept {
  return coefficients.system_count();
}

inline bool rhs_batch_size_is_compatible(
    const TridiagonalShared&,
    const std::size_t) noexcept {
  return true;
}

inline bool rhs_batch_size_is_compatible(
    const TridiagonalShiftedDiagonal& coefficients,
    const std::size_t batch_size) noexcept {
  return coefficients.system_count() == batch_size;
}

inline bool rhs_batch_size_is_compatible(
    const TridiagonalSystemDiagonal& coefficients,
    const std::size_t batch_size) noexcept {
  return coefficients.system_count() == batch_size;
}

inline bool rhs_batch_size_is_compatible(
    const TridiagonalBatch& coefficients,
    const std::size_t batch_size) noexcept {
  return coefficients.system_count() == batch_size;
}

inline bool rhs_batch_size_is_compatible(
    const SharedPreparedOperator&,
    const std::size_t) noexcept {
  return true;
}

inline bool rhs_batch_size_is_compatible(
    const PreparedOperatorBatch& prepared,
    const std::size_t batch_size) noexcept {
  return prepared.batch_size() == batch_size;
}

}  // namespace dtdma::detail
