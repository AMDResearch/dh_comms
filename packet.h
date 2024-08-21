#pragma once

namespace dh_comms {

    struct packet {
        uint32_t is_first : 1;
        uint32_t is_last  : 1;
        uint32_t thread_x : 10;
        uint32_t thread_y : 10;
        uint32_t thread_z : 10;
        uint16_t wg_x;
        uint16_t wg_y;
        uint16_t wg_z;
        uint16_t value; // TODO: current payload structure for debugging
        char payload[52];
    };

    __device__ void fill_packet(packet& p, bool is_first, bool is_last, uint16_t value);
    std::string packet_str(const packet& p);

    constexpr size_t bytes_per_packet = 64;

} // namespace dh_comms