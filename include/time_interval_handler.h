#pragma once
#include "message_handlers.h"

namespace dh_comms {
struct time_interval {
  uint64_t start;
  uint64_t stop;
};

class time_interval_handler_t : public message_handler_base {
public:
  time_interval_handler_t(bool verbose);
  time_interval_handler_t(const time_interval_handler_t &) = default;
  virtual ~time_interval_handler_t() = default;
  virtual bool handle(const message_t &message) override;
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