#pragma once
#include "message_handlers.h"

#include <map>
#include <set>
#include <vector>

namespace dh_comms {
//! The conflict_set class is used in the analysis of LDS bank conflicts. LDS memory
//! is partitioned into 32 banks of 4-byte dwords each. If two lanes in a wavefront
//! access two different addresses on the same bank simultaneously, that is a bank
//! conflict: those accesses have to be serialized. But: the memory accesses for
//! all 64 lanes in a wavefront are executed in phases, and which lanes are active
//! in which phase depends on the size of the read/write. For instance, if we read
//! or write floats (4 bytes/a dword), then lanes 0..31 are executed in one phase,
//! and lanes 32..63 are executed in another phase. Only lanes that are executed
//! in the same phase can cause a bank conflict, so we'd never see a bank conflict
//! for float reads or writes between e.g. lane 0 and lane 48, even if these lanes
//! access different addresses on the same bank. A conflict set is a set of lanes
//! such that any two of them may cause a bank conflict.
//!
//! For accesses of size 1, 2, or 4 bytes, the conflict sets are {0..31} and {32..63}.
//!
//! For accesses of size 8 bytes, the conflict sets are {0..15}, {16,31}, {32..47},
//! and {48..63}.
//!
//! For accesses of size 16 bytes, the conflict sets are {0..3, 20..23},
//! {4..7, 16..19}, {8..11, 28..31}, {12..15, 24..27}, {32..35, 52..55},
//! {36..39, 48..51}, {40..43, 60..63} and {44..476, 56..59}. Note the the
//! eight lanes in each conflict group are not consecutive in this case; they
//! consist of two subsets of 4 consecutive lanes each, but there is a gap between
//! the subset.
//!
//! For an access to LDS, the number of bank conflicts for that access is the sum
//! of the bank conflicts for the corresponding conflict sets. For a conflict set,
//! the number of bank conflict is the maximum of the conflicts for all banks minus
//! one. In other words, it is the number of _additional_ memory requests that have
//! to be issued for the conflict set compared to an optimal access, where each bank
//! is accessed once.

class conflict_set {
public:
  conflict_set(const std::vector<std::pair<std::size_t, std::size_t>> &fl_pairs);
  bool register_access(size_t lane, uint64_t address);
  size_t bank_conflict_count() const;
  void clear();

private:
  std::set<size_t> lanes;
  std::vector<std::set<uint64_t>> banks;
};

//! The memory_analysis_handler_t class handles messages with address data. If these
//! are global memory addresses, the total number of cache lines needed to access the
//! addresses for all active lanes in the wavefront is compared to the optimal number of
//! cache lines needed when all addresses are consecutive. If the memory accesses are
//! LDS accesses, the number of bank conflicts for the accesses are counted.
class memory_analysis_handler_t : public message_handler_base {
public:
  memory_analysis_handler_t(bool verbose);
  memory_analysis_handler_t(const memory_analysis_handler_t &) = default;
  virtual ~memory_analysis_handler_t() = default;
  virtual bool handle(const message_t &message) override;
  virtual bool handle(const message_t &message, const std::string &kernel_name, kernelDB::kernelDB &kdb) override;
  virtual void report() override;
  virtual void report(const std::string &kernel_name, kernelDB::kernelDB &kdb) override;
  virtual void clear() override;

private:
  bool handle_bank_conflict_analysis(const message_t &message);
  bool handle_cache_line_count_analysis(const message_t &message);
  void report_cache_line_use();
  void report_bank_conflicts();

private:
  struct counts_t {
    size_t no_accesses = 0;
    size_t no_conflicts = 0;
  };
  using s2c_t = std::map<uint8_t, counts_t>;       //!< maps size of the LDS reads/writes to bank conflict counts
  using rw2s2c_t = std::map<uint8_t, s2c_t>;       //!< maps operation type (read/write) to s2c_t;
  using l2rw2s2c_t = std::map<uint32_t, rw2s2c_t>; //!< maps location identifier to rw2s2c_t
  //! Bank conflict counts are tracked for each location, memory access type (read or write), and size of the
  //! access. For each combination, we keep track of how often there was an access, and how many bank conflicts
  //! occurred.
  l2rw2s2c_t bank_conflict_counts;

  //! Maps each of the supported memory access sizes to the conflict sets for that size
  std::map<std::size_t, std::vector<conflict_set>> conflict_sets;

  struct memory_accesses_t {
    size_t no_accesses = 0;
    uint16_t ir_access_size = 0;
    uint16_t isa_access_size = 0;
    uint8_t rw_kind = 0;
    std::string isa_instruction;

    bool operator==(const memory_accesses_t &other) {
      return ir_access_size == other.ir_access_size and isa_access_size == other.isa_access_size and
             rw_kind == other.rw_kind;
    }
    bool operator!=(const memory_accesses_t &other) { return not operator==(other); };
  };

  struct lds_accesses_t : memory_accesses_t {
    size_t no_bank_conflicts = 0;
  };

  struct global_accesses_t : memory_accesses_t {
    size_t min_cache_lines_needed = 0;
    size_t no_cache_lines_used = 0;
  };

  // We may have multiple memory accesses for a source code location, e.g. in expressions like "a += b;"
  // define a hierarchical data structure to map file/line/column to a set of memory accesses with associated data.
  template <typename T> using memory_access_set_t = std::vector<T>;
  template <typename T> using col_access_t = std::map<uint16_t, memory_access_set_t<T>>;
  template <typename T> using line_col_access_t = std::map<uint16_t, col_access_t<T>>;
  template <typename T> using file_line_col_access_t = std::map<std::string, line_col_access_t<T>>;

  file_line_col_access_t<global_accesses_t> global_accesses;
  file_line_col_access_t<lds_accesses_t> lds_accesses;

  kernelDB::kernelDB *kdb_p = nullptr;
  std::string kernel_name = "";
  bool verbose_;
  const std::map<uint8_t, const char *> rw2str_map;

public:
  struct access_size_and_type {
    uint16_t size;
    uint8_t access_type;
  };

private:
  const std::map<std::string, access_size_and_type> instr_size_map;
};
} // namespace dh_comms
