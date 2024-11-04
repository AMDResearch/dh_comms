#include "memory_analysis_handler.h"

#include <cassert>

namespace {
constexpr size_t no_banks = 32;
}

namespace dh_comms {

conflict_set::conflict_set(const std::vector<std::pair<std::size_t, std::size_t>> &fl_pairs)
    : lanes(),
      banks(std::vector<std::set<uint64_t>>(32)) {
  for (const auto &fl_pair : fl_pairs) {
    assert(fl_pair.first < fl_pair.second);
    for (std::size_t i = fl_pair.first; i != fl_pair.second; ++i) {
      lanes.insert(i);
    }
  }
}
bool conflict_set::register_access(size_t lane, uint64_t address) {
  if (lanes.find(lane) == lanes.end()) { // lane is not in this conflict set
    return false;
  }
  uint64_t dword = address / sizeof(uint32_t);
  size_t bank = dword % no_banks;
  banks[bank].insert(dword);
  return true;
}

size_t conflict_set::bank_conflict_count() const {
  size_t max_different_dwords_per_bank = 1;
  for (const auto &bank : banks) {
    max_different_dwords_per_bank = std::max(max_different_dwords_per_bank, bank.size());
  }
  return max_different_dwords_per_bank - 1;
}

void conflict_set::clear() {
  for (auto &bank : banks) {
    bank.clear();
  }
}

memory_analysis_handler_t::memory_analysis_handler_t(bool verbose)
    : conflict_sets(),
      verbose_(verbose) {
  conflict_sets.insert({1, std::vector<conflict_set>{
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{0, 32}}},
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{32, 64}}},
                           }});
  conflict_sets.insert({2, std::vector<conflict_set>{
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{0, 32}}},
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{32, 64}}},
                           }});
  conflict_sets.insert({4, std::vector<conflict_set>{
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{0, 32}}},
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{32, 64}}},
                           }});
  conflict_sets.insert({8, std::vector<conflict_set>{
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{0, 16}}},
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{16, 32}}},
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{32, 48}}},
                               conflict_set{std::vector<std::pair<size_t, size_t>>{{48, 64}}},
                           }});
  conflict_sets.insert({16, std::vector<conflict_set>{
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{0, 4}, {20, 24}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{4, 8}, {16, 20}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{8, 12}, {28, 32}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{12, 16}, {24, 28}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{32, 36}, {52, 56}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{36, 40}, {48, 52}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{40, 44}, {60, 64}}},
                                conflict_set{std::vector<std::pair<size_t, size_t>>{{44, 48}, {56, 60}}},
                            }});
}

bool memory_analysis_handler_t::handle(const message_t &message) {
  if (message.wave_header().user_type != message_type::address) {
    if (verbose_) {
      printf("memory_heatmap: skipping message with user type 0x%x\n", message.wave_header().user_type);
    }
    return false;
  }

  uint8_t mspace = (message.wave_header().user_data >> 2) & 0xf;
  switch (mspace) {
  case address_space::flat:
    break;
  case address_space::global:
    break;
  case address_space::gds:
    break;
  case address_space::shared:
    return handle_bank_conflict_analysis(message);
    break;
  case address_space::scratch:
    break;
  case address_space::undefined:
    break;
  default:
    break;
  }

  return false;
}

bool memory_analysis_handler_t::handle_bank_conflict_analysis(const message_t &message) {
  assert(message.data_item_size() == sizeof(uint64_t));
  auto lane_ids_of_active_lanes = get_lane_ids_of_active_lanes(message.wave_header());
  assert(message.no_data_items() == lane_ids_of_active_lanes.size());
  uint8_t rw_kind = message.wave_header().user_data & 0b11;
  uint16_t data_size = (message.wave_header().user_data >> 6) & 0xffff;
  if (conflict_sets.find(data_size) == conflict_sets.end()) {
    printf("bank conflict handling of %u-byte accesses not supported\n", data_size);
    return false;
  }

  for (size_t i = 0; i != message.no_data_items(); ++i) {
    auto lane = lane_ids_of_active_lanes[i];
    uint64_t address = *(const uint64_t *)message.data_item(i);
    assert(address % data_size == 0); // we only handle naturally-aligned data
    for (auto &cs : conflict_sets[data_size]) {
      if (cs.register_access(lane, address)) {
        break;
      }
    }
  }

  size_t bank_conflict_count = 0;
  for (auto &cs : conflict_sets[data_size]) {
    bank_conflict_count += cs.bank_conflict_count();
    cs.clear();
  }

  if (verbose_) {
    std::string rw_string;
    switch (rw_kind) {
    case memory_access::undefined:
      rw_string = "undefined memory operation";
      break;
    case memory_access::read:
      rw_string = "read";
      break;
    case memory_access::write:
      rw_string = "write";
      break;
    case memory_access::read_write:
      rw_string = "read/write";
      break;
    default:
      rw_string = "invalid memory operation";
      break;
    }
    printf("location %u: %s of %u bytes/lane, execution mask = 0x%zx, %zu bank conflicts\n",
           message.wave_header().src_loc_idx, rw_string.c_str(), data_size, message.wave_header().exec,
           bank_conflict_count);
  }

  // TODO: add bank conflict count to aggregate data structure.

  return true;
}

void memory_analysis_handler_t::report() { printf("nothing to report yet, stay tuned...\n"); }

void memory_analysis_handler_t::clear() {}

} // namespace dh_comms