#pragma once
#include <cstddef>
#include <vector>
#include "packet.h"

namespace dh_comms
{
    class buffer
    {
    public:
        buffer(std::size_t packets_per_sub_buffer);
        ~buffer();
        buffer(const buffer &) = delete;
        buffer &operator=(const buffer &) = delete;

        void print_cu_to_index_map() const;
        void show_queues() const;

    private:
        std::size_t no_sub_buffers_; // Current implementation: 1 sub-buffer per CU
        std::size_t packets_per_sub_buffer_;
        std::vector<uint16_t> index_to_cu_map_; // maps indices 0..#CUs - 1 to 11 bits CU ids (XCC|SE|CU)
        std::vector<size_t> cu_to_index_map_;   // inverse of the above map

        // device memory. Currently assumining a single GPU device.
        // TODO: take multiple devices into account

        // single memory buffer. Each CU writes to a sub-buffer, i.e., a section within the buffer.
        // TODO: does host see changes to device memory during kernel execution? If not, we may have
        // to use host-pinned memory instead, which would slow down the writes from the device.
        void *packet_buffer_ = nullptr;
        // map that allows waves to determine which sub-buffer to write to, based on the XCC|SE|CU
        // on which they run.
        size_t *sub_buffer_sizes_;
        size_t *cu_to_index_map_d_;
        // flags used for synchronizing data access between multiple device threads, and between
        // device/host code
        uint8_t *atomic_flags_;
    };

    __device__ uint16_t get_cu_id();
    __device__ size_t cu_to_index_map_f(uint16_t cu_id);
    __device__ void test_constants_f();
    __device__ void submit_packet(const packet& p);

} // namespace dh_comms