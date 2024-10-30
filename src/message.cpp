#include "message.h"

#include <cassert>
#include <cstring>

namespace {
std::vector<dh_comms::lane_header_t> init_lane_headers(const char *message_p,
                                                       const dh_comms::wave_header_t &wave_header) {
  const char *lane_header_p = message_p + sizeof(dh_comms::wave_header_t);
  size_t no_lane_headers = wave_header.has_lane_headers * wave_header.active_lane_count;
  std::vector<dh_comms::lane_header_t> lane_headers;
  lane_headers.reserve(no_lane_headers);
  for (std::size_t i = 0; i != no_lane_headers; ++i) {
    lane_headers.emplace_back(lane_header_p);
    lane_header_p += sizeof(dh_comms::lane_header_t);
  }
  return lane_headers;
}

std::vector<std::vector<char>> init_data(const char *message_p, const dh_comms::wave_header_t &wave_header) {
  const uint32_t *data_p = (const uint32_t *)(message_p + sizeof(dh_comms::wave_header_t) +
                                              wave_header.has_lane_headers * wave_header.active_lane_count *
                                                  sizeof(dh_comms::lane_header_t));
  size_t data_size = wave_header.data_size -
                     wave_header.has_lane_headers * wave_header.active_lane_count * sizeof(dh_comms::lane_header_t);
  size_t no_data_entries = (data_size == 0 ? 0 : (wave_header.is_vector_message ? wave_header.active_lane_count : 1));
  size_t dword_count = no_data_entries == 0 ? 0 : data_size / no_data_entries / sizeof(uint32_t);
  std::vector<std::vector<char>> data(no_data_entries, std::vector<char>(dword_count * sizeof(uint32_t)));
  for (size_t dword = 0; dword != dword_count; ++dword) {
    for (size_t entry = 0; entry != no_data_entries; ++entry) {
      memcpy((char *)&data[entry][dword * sizeof(uint32_t)], (char *)data_p++, sizeof(uint32_t));
    }
  }

  return data;
}
} // namespace

namespace dh_comms {
message_t::message_t(const char *message_p)
    : wave_header_(message_p),
      lane_headers_(init_lane_headers(message_p, wave_header_)),
      data_(init_data(message_p, wave_header_)) {}

size_t message_t::size() const { return sizeof(wave_header_t) + wave_header_.data_size; }
} // namespace dh_comms