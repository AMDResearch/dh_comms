#pragma once
#include "message_handlers.h"

namespace dh_comms {

class memory_analysis_handler_t : public message_handler_base {
public:
  memory_analysis_handler_t(bool verbose);
  memory_analysis_handler_t(const memory_analysis_handler_t &) = default;
  virtual ~memory_analysis_handler_t() = default;
  virtual bool handle(const message_t &message) override;
  virtual void report() override;
  virtual void clear() override;

private:
  bool verbose_;
};
} // namespace dh_comms