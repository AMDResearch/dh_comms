#pragma once
#include "message_handlers.h"

#include <map>
#include <set>
#include <vector>

namespace dh_comms {

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

class memory_analysis_handler_t : public message_handler_base {
public:
  memory_analysis_handler_t(bool verbose);
  memory_analysis_handler_t(const memory_analysis_handler_t &) = default;
  virtual ~memory_analysis_handler_t() = default;
  virtual bool handle(const message_t &message) override;
  virtual void report() override;
  virtual void clear() override;

private:
  bool handle_bank_conflict_analysis(const message_t &message);

private:
  std::map<std::size_t, std::vector<conflict_set>> conflict_sets;
  bool verbose_;
};
} // namespace dh_comms