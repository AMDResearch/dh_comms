#include "data_headers.h"

#include <cstring>

namespace dh_comms {
wave_header_t::wave_header_t(const char *wave_header_p) { memcpy((char *)this, wave_header_p, sizeof(wave_header_t)); }

lane_header_t::lane_header_t(const char *lane_header_p) { memcpy((char *)this, lane_header_p, sizeof(lane_header_t)); }

//! Returns a vector with the lane ids of the active lanes based on the execution mask in the wave header.
std::vector<size_t> get_lane_ids_of_active_lanes(const wave_header_t &wave_header) {
  std::vector<size_t> lane_ids;
  std::size_t lane_mask = 1;
  for (std::size_t lane_id = 0; lane_id != 64; ++lane_id) {
    if (wave_header.exec & lane_mask) {
      lane_ids.push_back(lane_id);
    }
    lane_mask <<= 1;
  }
  return lane_ids;
}

} // namespace dh_comms