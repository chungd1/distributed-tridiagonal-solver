#pragma once

#include <cstddef>

namespace dtdma {

[[nodiscard]] constexpr std::size_t canonical_index(
    const std::size_t row,
    const std::size_t system,
    const std::size_t batch_size) noexcept {
  return row * batch_size + system;
}

}  // namespace dtdma
