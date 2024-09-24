#pragma once
#include <cstddef>
#include <vector>
#include <thread>

#include "data_headers.h"
#include "hip_utils.h"

namespace dh_comms
{
    class buffer
    {
    public:
        buffer(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity, std::size_t no_host_threads = 1);
        ~buffer();
        buffer(const buffer &) = delete;
        buffer &operator=(const buffer &) = delete;

    private:
        void process_sub_buffers(std::size_t first, std::size_t last);
        std::vector<std::thread> init_host_threads(std::size_t no_host_threads);

    private:
        std::size_t no_sub_buffers_;
        std::size_t sub_buffer_capacity_;

        // device memory. Currently assumining a single GPU device.
        // TODO: take multiple devices into account

        // single memory buffer. Each workgroup writes to some sub-buffer, i.e., a section within the buffer.
        char *buffer_ = nullptr;
        // map that allows waves to determine which sub-buffer to write to, based on the XCC|SE|CU
        // on which they run.
        size_t *sub_buffer_sizes_;
        // flags used for synchronizing data access between multiple device threads, and between
        // device/host code
        uint8_t *atomic_flags_;
        volatile bool teardown_;
        std::vector<std::thread> sub_buffer_processors_;
    };

    __device__ size_t no_sub_buffers_f();
    __device__ size_t sub_buffer_capacity_f();
    __device__ char *buffer_f();
    __device__ size_t *sub_buffer_sizes_f();
    __device__ uint8_t *atomic_flags_f();

    __device__ size_t get_sub_buffer_idx();
    __device__ void wave_acquire(size_t sub_buf_idx, unsigned int active_lane_id);
    __device__ void wave_release(size_t sub_buf_idx, unsigned int active_lane_id);
    __device__ void wave_signal_host(size_t sub_buf_idx, unsigned int active_lane_id, bool return_control);

    template <typename message_t>
    __device__ inline void submit_message(const message_t &message, uint32_t user_type)
    {
        unsigned int active_lane_id = __active_lane_id();
        size_t sub_buf_idx = get_sub_buffer_idx();
        uint64_t exec = __builtin_amdgcn_read_exec();
        unsigned int active_lane_count = __active_lane_count(exec);
        size_t *sub_buffer_sizes = sub_buffer_sizes_f();
        size_t sub_buffer_capacity = sub_buffer_capacity_f();
        void *buffer = buffer_f();
        uint64_t data_size = active_lane_count * (sizeof(lane_header_t) + 4 * ((sizeof(message_t) + 3) / 4));

        wave_acquire(sub_buf_idx, active_lane_id);

        // first write the wave header; only one wave takes care of that
        if (active_lane_id == 0)
        {
            // start with writing the single wave header, but make sure first there is sufficient space
            // we'll split up the message in dwords (4 bytes) for improved writing speed, so round up the message
            // size to the nearest multiple of 4 bytes
            size_t current_size;
            while (((current_size = sub_buffer_sizes[sub_buf_idx]) + sizeof(wave_header_t) + data_size) > sub_buffer_capacity)
            {
                bool return_control = true;
                wave_signal_host(sub_buf_idx, active_lane_id, return_control);
            }
            wave_header_t wave_header(exec, data_size, active_lane_count, user_type);
            size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size;
            wave_header_t *wave_header_p = (wave_header_t *)(&((char *)buffer)[byte_offset]);
            *wave_header_p = wave_header;
            sub_buffer_sizes[sub_buf_idx] += sizeof(wave_header_t);
            // printf("sizeof(wave_header_t) = %zu\n", sizeof(wave_header_t));
            // printf("sizeof(lane_header_t) = %zu\n", sizeof(lane_header_t));
            // printf("message size = %zu\n", sizeof(message_t));
            // printf("data size (including lane header) = %zu\n", data_size);
        }

        // write the lane headers for the active lanes
        size_t current_size = sub_buffer_sizes[sub_buf_idx];
        lane_header_t lane_header;
        size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size + active_lane_id * sizeof(lane_header);
        lane_header_t *lane_header_p = (lane_header_t *)(&((char *)buffer)[byte_offset]);
        // printf("lane_id = %u, active_lane_id = %u, sub-buffer size = %zu, writing to offset %zu\n", __lane_id(), __active_lane_id(), current_size, byte_offset);
        *lane_header_p = lane_header;
        byte_offset += active_lane_count * sizeof(lane_header);

        // write the message one dword at a time in a coalesced fashion
        size_t remaining_bytes = sizeof(message);
        char *src = (char *)&message;
        char *dst = &((char *)buffer)[byte_offset];
        while (remaining_bytes)
        {
            size_t size = min(remaining_bytes, 4);
            memcpy(dst, src, size);
            src += 4;
            dst += 4 * active_lane_count;
            remaining_bytes -= size;
        }

        // update sub-buffer size and release lock
        if (active_lane_id == 0)
        {
            sub_buffer_sizes[sub_buf_idx] += data_size;
        }
        wave_release(sub_buf_idx, active_lane_id);
    }

} // namespace dh_comms