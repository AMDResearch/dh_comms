#pragma once

#include "dh_comms.h"

// device functions
namespace dh_comms
{
    __device__ inline size_t get_sub_buffer_idx(size_t no_sub_buffers)
    {
        size_t grid_dim_x_m = gridDim.x % no_sub_buffers;
        size_t grid_dim_y_m = gridDim.y % no_sub_buffers;
        size_t grid_dim_xy_m = (grid_dim_x_m * grid_dim_y_m) % no_sub_buffers;
        size_t block_id_z_grid_dim_xy_m = (blockIdx.z * grid_dim_xy_m) % no_sub_buffers;

        size_t block_id_y_grid_dim_x_m = (blockIdx.y * grid_dim_x_m) % no_sub_buffers;

        size_t block_id_x_m = blockIdx.x % no_sub_buffers;

        return (block_id_z_grid_dim_xy_m + block_id_y_grid_dim_x_m + block_id_x_m) % no_sub_buffers;
    }

    __device__ inline void wave_acquire(uint8_t *atomic_flags, size_t sub_buf_idx, unsigned int active_lane_id)
    {
        if (active_lane_id == 0)
        {
            uint8_t expected = 0;
            uint8_t desired = 1;
            bool weak = false;
            while (not __atomic_compare_exchange(&atomic_flags[sub_buf_idx], &expected, &desired, weak,
                                                 __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            {
                // __atomic_compare_exchange returns the value at the address (first argument) in
                // the second argument (expected), so we need to reset it in this spin-loop
                expected = 0;
            }
        }
    }

    __device__ inline void wave_release(uint8_t *atomic_flags, size_t sub_buf_idx, unsigned int active_lane_id)
    {
        if (active_lane_id == 0)
        {
            uint8_t flag_value = 0;
            __atomic_store(&atomic_flags[sub_buf_idx], &flag_value, __ATOMIC_RELEASE);
        }
    }

    __device__ inline void wave_signal_host(uint8_t *atomic_flags, size_t sub_buf_idx, unsigned int active_lane_id, bool return_control)
    {
        if (active_lane_id == 0)
        {
            uint8_t flag_value = return_control ? 3 : 2;
            // printf("[Device] signalling buffer full for idx %lu\n", sub_buf_idx);
            __atomic_store(&atomic_flags[sub_buf_idx], &flag_value, __ATOMIC_RELEASE);
            // when host code sees a flag value > 1, it starts processing the sub-buffer that
            // we indicated to be full. Once the host is done, it subtracts 2 from the flag,
            // i.e., if return_control is true, the flag will be 1 after the host is done,
            // and this wave will wait for that to take control. Otherwise, if return_control
            // is false, the host will set the flag to 0, and any wave that wants to write
            // to the sub-buffer can get control by calling wave_acquire()
            if (return_control)
            {
                // spin until host sets flag to 1, indicating it returns control to this wave
                while (__atomic_load_n(&atomic_flags[sub_buf_idx], __ATOMIC_ACQUIRE) != 1)
                {
                }
            }
        }
    }

    template <typename message_t>
    __device__ inline void v_submit_message(dh_comms_resources *rsrc, const message_t &message, uint32_t user_type)
    {
        unsigned int active_lane_id = __active_lane_id();
        size_t sub_buf_idx = get_sub_buffer_idx(rsrc->no_sub_buffers_);
        uint64_t exec = __builtin_amdgcn_read_exec();
        unsigned int active_lane_count = __active_lane_count(exec);
        size_t *sub_buffer_sizes = rsrc->sub_buffer_sizes_;
        size_t sub_buffer_capacity = rsrc->sub_buffer_capacity_;
        void *buffer = rsrc->buffer_;
        uint64_t data_size = active_lane_count * (sizeof(lane_header_t) + 4 * ((sizeof(message_t) + 3) / 4));

        wave_acquire(rsrc->atomic_flags_, sub_buf_idx, active_lane_id);
        if (sizeof(wave_header_t) + data_size > sub_buffer_capacity)
        {
            *rsrc->error_bits_ |= 1;
            wave_release(rsrc->atomic_flags_, sub_buf_idx, active_lane_id);
            return;
        }
        size_t current_size = sub_buffer_sizes[sub_buf_idx];
        if (current_size + sizeof(wave_header_t) + data_size > sub_buffer_capacity)
        {
            bool return_control = true;
            wave_signal_host(rsrc->atomic_flags_, sub_buf_idx, active_lane_id, return_control);
            // the host clears the sub-buffer and resets its size to 0. Since it
            // returns control to us, no other wave can have changed the sub -buffer
            // size after wave_signal_host returns. So instead of re-reading the size
            // from memory, we can deduce that it must be 0. This saves a slow memory
            // read.
            current_size = 0;
        }
        // first write the wave header; only one wave takes care of that
        if (active_lane_id == 0)
        {
            // start with writing the single wave header, but make sure first there is sufficient space
            // we'll split up the message in dwords (4 bytes) for improved writing speed, so round up the message
            // size to the nearest multiple of 4 bytes
            wave_header_t wave_header(exec, data_size, active_lane_count, user_type);
            size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size;
            wave_header_t *wave_header_p = (wave_header_t *)(&((char *)buffer)[byte_offset]);
            *wave_header_p = wave_header;
        }

        // write the lane headers for the active lanes
        size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size + sizeof(wave_header_t) + active_lane_id * sizeof(lane_header_t);
        lane_header_t *lane_header_p = (lane_header_t *)(&((char *)buffer)[byte_offset]);
        lane_header_t lane_header;
        *lane_header_p = lane_header;
        byte_offset += active_lane_count * sizeof(lane_header_t);

        // write the message one dword at a time in a coalesced fashion
        size_t remaining_bytes = sizeof(message);
        char *src = (char *)&message;
        char *dst = &((char *)buffer)[byte_offset];
        while (remaining_bytes)
        {
            size_t size = min(remaining_bytes, sizeof(uint32_t));
            memcpy(dst, src, size);
            src += sizeof(uint32_t);
            dst += active_lane_count * sizeof(uint32_t);
            remaining_bytes -= size;
        }

        // update sub-buffer size and release lock
        if (active_lane_id == 0)
        {
            sub_buffer_sizes[sub_buf_idx] += sizeof(wave_header_t) + data_size;
        }
        wave_release(rsrc->atomic_flags_, sub_buf_idx, active_lane_id);
    }

} // namespace dh_comms