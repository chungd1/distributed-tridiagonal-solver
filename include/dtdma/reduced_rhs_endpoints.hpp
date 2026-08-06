#pragma once

#include "dtdma/reduced_rhs_batch.hpp"
#include "dtdma/scalar.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace dtdma {

class ReducedRhsEndpoints {
 public:
  explicit ReducedRhsEndpoints(const std::size_t batch_size)
      : batch_size_(batch_size),
        element_count_(checked_element_count(batch_size)),
        endpoints_(element_count_) {}

  [[nodiscard]] std::size_t batch_size() const noexcept { return batch_size_; }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return element_count_;
  }

  [[nodiscard]] Scalar& endpoint(const std::size_t system,
                                 const std::size_t endpoint) {
    return endpoints_[checked_index(system, endpoint)];
  }

  [[nodiscard]] const Scalar& endpoint(const std::size_t system,
                                       const std::size_t endpoint) const {
    return endpoints_[checked_index(system, endpoint)];
  }

  [[nodiscard]] std::span<Scalar> endpoints() noexcept { return endpoints_; }
  [[nodiscard]] std::span<const Scalar> endpoints() const noexcept {
    return endpoints_;
  }

 private:
  static constexpr std::size_t endpoint_count = 2;

  [[nodiscard]] static std::size_t checked_element_count(
      const std::size_t batch_size) {
    if (batch_size == 0) {
      throw std::invalid_argument("batch size must be greater than zero");
    }
    if (batch_size >
        std::numeric_limits<std::size_t>::max() / endpoint_count) {
      throw std::length_error("reduced RHS endpoint count overflows");
    }
    return endpoint_count * batch_size;
  }

  [[nodiscard]] std::size_t checked_index(
      const std::size_t system,
      const std::size_t endpoint) const {
    if (system >= batch_size_) {
      throw std::out_of_range("system index is out of range");
    }
    if (endpoint >= endpoint_count) {
      throw std::out_of_range("endpoint index is out of range");
    }
    return endpoint * batch_size_ + system;
  }

  std::size_t batch_size_;
  std::size_t element_count_;
  std::vector<Scalar> endpoints_;
};

inline void extract_reduced_rhs_endpoints(
    const ReducedRhsBatch& working,
    ReducedRhsEndpoints& endpoints) {
  if (working.row_count() < 3) {
    throw std::invalid_argument(
        "endpoint extraction requires at least three rows");
  }
  if (endpoints.batch_size() != working.batch_size()) {
    throw std::invalid_argument(
        "RHS working batch and endpoint dimensions must match");
  }

  const std::size_t last_row = working.row_count() - 1;
  for (std::size_t system = 0; system < working.batch_size(); ++system) {
    endpoints.endpoint(system, 0) = working.rhs(0, system);
    endpoints.endpoint(system, 1) = working.rhs(last_row, system);
  }
}

}  // namespace dtdma
