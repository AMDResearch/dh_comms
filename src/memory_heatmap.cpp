#include "memory_heatmap.h"

#include "data_headers.h"
#include "message.h"

#include <cassert>
#include <cstdio>
#include <vector>

namespace dh_comms {
memory_heatmap_t::memory_heatmap_t(size_t page_size, bool verbose)
    : verbose_(verbose),
      page_size_(page_size) {}

bool memory_heatmap_t::handle(const message_t &message) {
  if ((message_type)message.wave_header().user_type != message_type::address) {
    if (verbose_) {
      printf("memory_heatmap: skipping message with user type 0x%x\n", message.wave_header().user_type);
    }
    return false;
  }
  assert(message.data_item_size() == sizeof(uint64_t));
  for (size_t i = 0; i != message.no_data_items(); ++i) {
    uint64_t address = *(const uint64_t *)message.data_item(i);
    // map address to lowest address in page and update page count
    address /= page_size_;
    address *= page_size_;
    ++page_counts_[address];
    if (verbose_) {
      printf("memory_heatmap: added address 0x%lx to map\n", address);
    }
  }
  return true;
}

void memory_heatmap_t::report() {
  if (page_counts_.size() != 0) {
    printf("memory heatmap report:\n\tpage size = %lu\n", page_size_);
  }
  for (const auto &[first_page_address, count] : page_counts_) {
    auto last_page_address = first_page_address + page_size_ - 1;
    printf("\tpage [%016lx:%016lx] %12lu accesses\n", first_page_address, last_page_address, count);
  }
}

void memory_heatmap_t::clear() { page_counts_.clear(); }

} // namespace dh_comms