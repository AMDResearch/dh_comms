#include "memory_analysis_handler.h"

#include <cassert>

namespace {
void get_next_lane_id(uint64_t exec, size_t &lane_id) {
  uint64_t mask;
  if (lane_id > 63) {
    mask = 1;
    lane_id = 0;
  } else {
    ++lane_id;
    mask = 1 << lane_id;
  }
  while (not(exec & mask) and lane_id < 64) {
    ++lane_id;
    mask <<= 1;
  }
  assert(lane_id < 64);
}
} // namespace

namespace dh_comms {

memory_analysis_handler_t::memory_analysis_handler_t(bool verbose)
    : verbose_(verbose) {}

bool memory_analysis_handler_t::handle(const message_t &message) {
  if (message.wave_header().user_type != message_type::address) {
    if (verbose_) {
      printf("memory_heatmap: skipping message with user type 0x%x\n", message.wave_header().user_type);
    }
    return false;
  }
  assert(message.data_item_size() == sizeof(uint64_t));
  uint64_t exec = message.wave_header().exec;
  size_t lane_id = 64;
  for (size_t i = 0; i != message.no_data_items(); ++i) {
    get_next_lane_id(exec, lane_id);
    uint64_t address = *(const uint64_t *)message.data_item(i);
    if (verbose_) {
      printf("processing lane %zu: address 0x%lu\n", lane_id, address);
    }
  }
  return true;
}

void memory_analysis_handler_t::report() { printf("nothing to report yet, stay tuned...\n"); }

void memory_analysis_handler_t::clear() {}

} // namespace dh_comms