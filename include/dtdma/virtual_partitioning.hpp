#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace dtdma {

class VirtualPartitioning {
 public:
  struct Partition {
    std::size_t rank;
    std::size_t begin_row;
    std::size_t end_row;
    std::size_t local_row_count;
  };

  VirtualPartitioning(const std::size_t global_row_count,
                      const std::size_t partition_count)
      : global_row_count_(global_row_count),
        partition_count_(partition_count),
        base_row_count_(checked_base_row_count(global_row_count,
                                               partition_count)),
        remainder_(global_row_count % partition_count) {}

  [[nodiscard]] std::size_t global_row_count() const noexcept {
    return global_row_count_;
  }
  [[nodiscard]] std::size_t partition_count() const noexcept {
    return partition_count_;
  }

  [[nodiscard]] Partition partition(const std::size_t rank) const {
    if (rank >= partition_count_) {
      throw std::out_of_range("virtual rank index is out of range");
    }
    const std::size_t local_row_count =
        base_row_count_ + (rank < remainder_ ? 1 : 0);
    const std::size_t begin_row =
        rank * base_row_count_ + std::min(rank, remainder_);
    return Partition{rank, begin_row, begin_row + local_row_count,
                     local_row_count};
  }

 private:
  [[nodiscard]] static std::size_t checked_base_row_count(
      const std::size_t global_row_count,
      const std::size_t partition_count) {
    if (partition_count < 1 || partition_count > 4) {
      throw std::invalid_argument(
          "virtual partition count must be between one and four");
    }
    if (global_row_count / partition_count < 3) {
      throw std::invalid_argument(
          "each virtual partition requires at least three rows");
    }
    return global_row_count / partition_count;
  }

  std::size_t global_row_count_;
  std::size_t partition_count_;
  std::size_t base_row_count_;
  std::size_t remainder_;
};

}  // namespace dtdma
