#pragma once
#include <cstddef>
#include <vector>
#include <thread>

#include "packet.h"

namespace dh_comms
{
    class buffer
    {
    public:
        buffer(std::size_t no_sub_buffers, std::size_t packets_per_sub_buffer);
        ~buffer();
        buffer(const buffer &) = delete;
        buffer &operator=(const buffer &) = delete;

        void show_queues() const;

    private:
        void process_buffer();

    private:
        std::size_t no_sub_buffers_; // Current implementation: 1 sub-buffer per CU
        std::size_t packets_per_sub_buffer_;

        // device memory. Currently assumining a single GPU device.
        // TODO: take multiple devices into account

        // single memory buffer. Each workgroup writes to some sub-buffer, i.e., a section within the buffer.
        void *packet_buffer_ = nullptr;
        // map that allows waves to determine which sub-buffer to write to, based on the XCC|SE|CU
        // on which they run.
        size_t *sub_buffer_sizes_;
        // flags used for synchronizing data access between multiple device threads, and between
        // device/host code
        uint8_t *atomic_flags_;
        volatile bool teardown_;
        std::thread buffer_processor_;
    };

    __device__ uint16_t get_cu_id();
    __device__ size_t cu_to_index_map_f(uint16_t cu_id);
    __device__ void test_constants_f();
    __device__ void submit_packet(const packet& p);

} // namespace dh_comms