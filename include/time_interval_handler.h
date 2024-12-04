#pragma once
#include "message_handlers.h"

namespace dh_comms {
struct time_interval {
  uint64_t start;
  uint64_t stop;
};

//! The time_interval_handler class processes time interval messages. It Keeps
//! track of the sum of the time covered by the messages as well as the total
//! elapsed time between the earliest start time in any message and the latest
//! stop time in any message.
class time_interval_handler_t : public message_handler_base {
public:
  time_interval_handler_t(bool verbose);
  time_interval_handler_t(const time_interval_handler_t &) = default;
  virtual ~time_interval_handler_t() = default;
  virtual bool handle(const message_t &message) override;
  virtual bool handle(const message_t &message, const std::string& kernel_name, kernelDB::kernelDB& kdb) override;
  virtual void report() override;
  virtual void clear() override;

private:
  uint64_t first_start_;
  uint64_t last_stop_;
  uint64_t total_time_;
  size_t no_intervals_;
  bool verbose_;
};
} // namespace dh_comms
