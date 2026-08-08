#pragma once

#include "dtdma/backward_coefficient_preparation.hpp"
#include "dtdma/forward_coefficient_preparation.hpp"
#include "dtdma/prepared_operator_batch.hpp"
#include "dtdma/prepared_operator_batch_shared_offdiagonals.hpp"
#include "dtdma/prepared_operator_shared.hpp"
#include "dtdma/tridiagonal_batch.hpp"
#include "dtdma/tridiagonal_shared.hpp"
#include "dtdma/tridiagonal_shifted_diagonal.hpp"
#include "dtdma/tridiagonal_system_diagonal.hpp"

#include <utility>

namespace dtdma {

inline PreparedOperatorShared prepare_operator(
    TridiagonalShared&& original) {
  PreparedOperatorShared prepared(original.row_count());
  prepare_forward_coefficients(original, prepared);
  prepare_backward_coefficients(original, prepared);
  prepared.retained_bands_ = std::move(original.storage_);
  return prepared;
}

inline PreparedOperatorBatchSharedOffdiagonals prepare_operator(
    TridiagonalShiftedDiagonal&& original) {
  PreparedOperatorBatchSharedOffdiagonals prepared(
      original.row_count(), original.system_count());
  prepare_forward_coefficients(original, prepared);
  prepare_backward_coefficients(original, prepared);
  prepared.retained_bands_ = std::move(original.storage_);
  return prepared;
}

inline PreparedOperatorBatchSharedOffdiagonals prepare_operator(
    TridiagonalSystemDiagonal&& original) {
  PreparedOperatorBatchSharedOffdiagonals prepared(
      original.row_count(), original.system_count());
  prepare_forward_coefficients(original, prepared);
  prepare_backward_coefficients(original, prepared);
  prepared.retained_bands_ = std::move(original.storage_);
  return prepared;
}

inline PreparedOperatorBatch prepare_operator(
    TridiagonalBatch&& original) {
  PreparedOperatorBatch prepared(original.row_count(),
                                  original.system_count());
  prepare_forward_coefficients(original, prepared);
  prepare_backward_coefficients(original, prepared);
  prepared.lower_ = std::move(original.lower_);
  prepared.upper_ = std::move(original.upper_);
  return prepared;
}

}  // namespace dtdma
