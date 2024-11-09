#include "memory_analysis_handler.h"

#include "hip_utils.h"
#include "utils.h"

#include <cassert>
#include <hip/hip_runtime.h>
#include <set>
#include <string>

namespace {
constexpr size_t no_banks = 32;

const uint8_t L2_cache_line_sizes[] = {
    0,   // unsupported archs
    64,  // gfx906
    64,  // gfx908
    128, // gfx90a
    128, // gfx940
    128, // gfx941
    128  // gfx942
};

} // namespace

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
    : bank_conflict_counts(),
      conflict_sets(),
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
  assert(message.data_item_size() == sizeof(uint64_t));
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
    return handle_cache_line_count_analysis(message);
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

const std::map<uint8_t, const char *> rw2str_map = {
    {memory_access::undefined, "unspecified memory operation"},
    {memory_access::read, "read"},
    {memory_access::write, "write"},
    {memory_access::read_write, "read/write"},
};

std::string rw2str(uint8_t rw_kind) {
  std::string rw_string;
  const auto &rw_s = rw2str_map.find(rw_kind);
  if (rw_s != rw2str_map.end()) {
    rw_string = rw_s->second;
  } else {
    rw_string = "[coding error: invalid encoding of memory operation type]";
  }
  return rw_string;
}

bool memory_analysis_handler_t::handle_cache_line_count_analysis(const message_t &message) {
  uint8_t L2_cache_line_size = L2_cache_line_sizes[message.wave_header().arch];
  if (L2_cache_line_size == 0) {
    if (verbose_) {
      printf("Memory analysis handler: message from unsupported GPU hardware, skipping.\n");
    }
    return false;
  }

  uint8_t rw_kind = message.wave_header().user_data & 0b11;
  uint16_t data_size = (message.wave_header().user_data >> 6) & 0xffff;
  size_t min_cache_lines_needed = (message.no_data_items() * data_size + L2_cache_line_size - 1) / L2_cache_line_size;
  std::set<uint64_t> cache_lines;
  for (size_t i = 0; i != message.no_data_items(); ++i) {
    // take into account that in odd cases, the memory access may stride more than a single cache line
    uint64_t first_byte_of_address = *(const uint64_t *)message.data_item(i);
    uint64_t last_byte_of_address = first_byte_of_address + data_size - 1;
    uint64_t first_cache_line_of_address = first_byte_of_address / L2_cache_line_size;
    uint64_t last_cache_line_of_address = last_byte_of_address / L2_cache_line_size;
    for (uint64_t cache_line = first_cache_line_of_address; cache_line <= last_cache_line_of_address; ++cache_line) {
      cache_lines.insert(cache_line);
    }
  }
  uint64_t cache_lines_used = cache_lines.size();

  if (verbose_) {
    std::string rw_string = rw2str(rw_kind);
    printf("location %u: global memory access\n"
           "\t%s of %u bytes/lane, minimum L2 cache lines required %zu, cache lines used %zu\n"
           "\texecution mask = %s\n",
           message.wave_header().src_loc_idx, rw_string.c_str(), data_size, min_cache_lines_needed, cache_lines_used,
           exec2binstr(message.wave_header().exec).c_str());
  }

  access_counts_t &counts = cache_line_use_counts[message.wave_header().src_loc_idx][rw_kind][data_size];
  ++counts.no_accesses;
  counts.min_cache_lines_needed += min_cache_lines_needed;
  counts.cache_lines_used += cache_lines_used;

  return true;
}

bool memory_analysis_handler_t::handle_bank_conflict_analysis(const message_t &message) {
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
    std::string rw_string = rw2str(rw_kind);
    printf("location %u: LDS access\n"
           "\t%s of %u bytes/lane, %zu bank conflicts\n"
           "\texecution mask = %s\n",
           message.wave_header().src_loc_idx, rw_string.c_str(), data_size, bank_conflict_count,
           exec2binstr(message.wave_header().exec).c_str());
  }

  counts_t &counts = bank_conflict_counts[message.wave_header().src_loc_idx][rw_kind][data_size];
  ++counts.no_accesses;
  counts.no_conflicts += bank_conflict_count;

  return true;
}

void memory_analysis_handler_t::report_bank_conflicts() {
  printf("\n=== Bank conflicts report =========================\n");
  bool found_conflicts = false;
  for (const auto &[loc, rw2s2c] : bank_conflict_counts) {
    bool found_conflicts_for_location = false;
    for (const auto &[rw_kind, s2c] : rw2s2c) {
      for (const auto &[size, counts] : s2c) {
        if (counts.no_conflicts != 0) {
          if (!found_conflicts_for_location) {
            printf("bank conflicts for location %u\n", loc);
            found_conflicts_for_location = true;
          }
          found_conflicts = true;
          std::string rw_string = rw2str(rw_kind);
          printf("\t%s of %u bytes/lane: executed %zu times, %zu total bank conflict(s)\n", rw_string.c_str(), size,
                 counts.no_accesses, counts.no_conflicts);
        }
      }
    }
  }
  if (!found_conflicts) {
    printf("No bank conflicts found\n");
  }
  printf("=== End of bank conflicts report ====================\n");
}

void memory_analysis_handler_t::report_cache_line_use() {
  printf("\n=== L2 cache line use report ======================\n");
  bool found_excess = false;
  for (const auto &[loc, rw2s2ac] : cache_line_use_counts) {
    bool found_excess_for_location = false;
    for (const auto &[rw_kind, s2ac] : rw2s2ac) {
      for (const auto &[size, counts] : s2ac) {
        if (counts.cache_lines_used > counts.min_cache_lines_needed != 0) {
          if (!found_excess_for_location) {
            printf("Excessive number of cache lines used for location %u\n", loc);
            found_excess_for_location = true;
          }
          found_excess = true;
          std::string rw_string = rw2str(rw_kind);
          printf("\t%s of %u bytes/lane: executed %zu times, %zu cache lines needed, %zu cache lines used\n",
                 rw_string.c_str(), size, counts.no_accesses, counts.min_cache_lines_needed, counts.cache_lines_used);
        }
      }
    }
  }
  if (!found_excess) {
    printf("No excess cache lines used for global memory accesses\n");
  }
  printf("=== End of L2 cache line use report ===============\n");
}

void memory_analysis_handler_t::report() {
  report_cache_line_use();
  report_bank_conflicts();
}

void memory_analysis_handler_t::clear() {
  bank_conflict_counts.clear();
  cache_line_use_counts.clear();
}
} // namespace dh_comms