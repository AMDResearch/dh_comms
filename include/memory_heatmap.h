#pragma once

#include "message_handlers.h"

#include <map>
namespace dh_comms {

//! The memory_heatmap_t class keeps track of how many accesses to each memory
//! page are done. Page size is configurable.
class memory_heatmap_t : public message_handler_base {
public:
  memory_heatmap_t(size_t page_size = 1024 * 1024, bool verbose = false);
  memory_heatmap_t(const memory_heatmap_t &) = default;
  virtual ~memory_heatmap_t() {};
  virtual bool handle(const message_t &message) override;
  virtual void report() override;
  virtual void clear() override;

private:
  bool verbose_;
  size_t page_size_;
  //! Maps the lowest address on each page to the number of accesses to the page.
  std::map<uint64_t, size_t> page_counts_;
};

} // namespace dh_comms