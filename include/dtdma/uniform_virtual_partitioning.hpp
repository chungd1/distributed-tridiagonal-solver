#pragma once

#include <cstddef>
#include <stdexcept>

namespace dtdma {

class UniformVirtualPartitioning {
 public:
  struct Partition {
    std::size_t virtual_rank_index;
    std::size_t local_row_begin;
    std::size_t local_row_end;
    std::size_t local_row_count;
  };

  UniformVirtualPartitioning(const std::size_t global_row_count,
                             const std::size_t virtual_rank_count)
      : global_row_count_(global_row_count),
        virtual_rank_count_(virtual_rank_count),
        local_row_count_(checked_local_row_count(global_row_count,
                                                 virtual_rank_count)) {}

  [[nodiscard]] std::size_t global_row_count() const noexcept {
    return global_row_count_;
  }
  [[nodiscard]] std::size_t virtual_rank_count() const noexcept {
    return virtual_rank_count_;
  }
  [[nodiscard]] std::size_t local_row_count() const noexcept {
    return local_row_count_;
  }

  [[nodiscard]] Partition partition(const std::size_t virtual_rank) const {
    if (virtual_rank >= virtual_rank_count_) {
      throw std::out_of_range("virtual rank index is out of range");
    }
    const std::size_t row_begin = virtual_rank * local_row_count_;
    return Partition{virtual_rank, row_begin, row_begin + local_row_count_,
                     local_row_count_};
  }

 private:
  [[nodiscard]] static std::size_t checked_local_row_count(
      const std::size_t global_row_count,
      const std::size_t virtual_rank_count) {
    if (virtual_rank_count < 1 || virtual_rank_count > 4) {
      throw std::invalid_argument(
          "virtual rank count must be between one and four");
    }
    if (global_row_count == 0 ||
        global_row_count % virtual_rank_count != 0) {
      throw std::invalid_argument(
          "global row count must divide uniformly across virtual ranks");
    }
    const std::size_t local_row_count =
        global_row_count / virtual_rank_count;
    if (local_row_count < 3) {
      throw std::invalid_argument(
          "each virtual partition requires at least three rows");
    }
    return local_row_count;
  }

  std::size_t global_row_count_;
  std::size_t virtual_rank_count_;
  std::size_t local_row_count_;
};

}  // namespace dtdma
