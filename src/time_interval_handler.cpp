#include "time_interval_handler.h"

#include <cassert>

namespace dh_comms {

time_interval_handler_t::time_interval_handler_t(bool verbose)
    : first_start_(0xffffffffffffffff),
      last_stop_(0),
      total_time_(0),
      no_intervals_(0),
      verbose_(verbose) {}

  
bool time_interval_handler_t::handle(const message_t &message, const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    // This if block is just to get the compiler to quick throwing errors for unused parameters
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    return handle(message);
}

bool time_interval_handler_t::handle(const message_t &message) {
  if (message.wave_header().user_type != message_type::time_interval) {
    if (verbose_) {
      printf("time_interval_handler: skipping message with user type 0x%x\n", message.wave_header().user_type);
    }
    return false;
  }
  assert(message.data_item_size() == sizeof(time_interval));
  for (size_t i = 0; i != message.no_data_items(); ++i) {
    time_interval ti = *(const time_interval *)message.data_item(i);
    assert(ti.start <= ti.stop);

    first_start_ = ti.start < first_start_ ? ti.start : first_start_;
    last_stop_ = ti.stop > last_stop_ ? ti.stop : last_stop_;
    total_time_ += ti.stop - ti.start;
    ++no_intervals_;
    if (verbose_) {
      printf("time_interval processed:\n");
      printf("\tstart = %lu\n", ti.start);
      printf("\tstop = %lu\n", ti.stop);
    }
  }
  return true;
}

void time_interval_handler_t::report() {
  if (no_intervals_ != 0) {
    double average_time = (double)total_time_ / no_intervals_;
    printf("time_interval report:\n");
    printf("\ttotal time for all %zu intervals: %lu\n", no_intervals_, total_time_);
    printf("\taverage time per interval: %.0f\n", average_time);
    printf("\tfirst start: %lu\n", first_start_);
    printf("\tlast stop: %lu\n", last_stop_);
    printf("\ttime from first start to last stop: %lu"
           "\t   (%f times the average interval time)\n",
           last_stop_ - first_start_, (last_stop_ - first_start_) / average_time);
  }
}

void time_interval_handler_t::clear() {
  first_start_ = 0;
  last_stop_ = 0;
  total_time_ = 0;
  no_intervals_ = 0;
}

} // namespace dh_comms
