#pragma once

//! @file

#include "dh_comms.h"

// device functions
namespace dh_comms
{
    // Have Doxygen skip implementation functions that aren't used directly in user code
    //! @cond

    // Returns the index of the sub-buffer to which the calling wave should write.
    __device__ inline size_t get_sub_buffer_idx(size_t no_sub_buffers)
    {
        // To balance the number of waves writing to the sub-buffers, this function
        // calculates the flattened workgroup/block index, and takes that value
        // modulo the number of sub-buffers.
        // So, we're computing (blockIdx.z * gridDim.y * grimDim.z +
        //                      blockIdx.y * gridDim.x +
        //                      blockIdx.x) % no_sub_buffers.
        // A slight problem here is that block dimensions and indices are
        // 32-bit values, so we'd need a 96-bit type to do our calculations
        // without risking overflow if we'd do the calculation as above.
        // Fortunately, the modulo operation has the property that
        // (a * b) % m = ((a % m) * (b % m)) % m and
        // (a + b) % m = ((a % m) + (b % m)) % m and
        // which means that we can apply the modulo operation on intermediate results,
        // and get the same final result without risking overflow.

        size_t grid_dim_x_m = gridDim.x % no_sub_buffers;
        size_t grid_dim_y_m = gridDim.y % no_sub_buffers;
        size_t grid_dim_xy_m = (grid_dim_x_m * grid_dim_y_m) % no_sub_buffers;
        size_t block_id_z_grid_dim_xy_m = (blockIdx.z * grid_dim_xy_m) % no_sub_buffers;

        size_t block_id_y_grid_dim_x_m = (blockIdx.y * grid_dim_x_m) % no_sub_buffers;

        size_t block_id_x_m = blockIdx.x % no_sub_buffers;

        return (block_id_z_grid_dim_xy_m + block_id_y_grid_dim_x_m + block_id_x_m) % no_sub_buffers;
    }

    // Have the calling wave spin on an atomic to get exclusive read/write access to a sub-buffer.
    __device__ inline void wave_acquire(uint8_t *atomic_flags, size_t sub_buf_idx, unsigned int active_lane_id)
    {
        if (active_lane_id == 0) // only one lane acquires the lock on behalf of the wave
        {
            uint8_t expected = 0; // 0 -> sub-buffer is not locked
            uint8_t desired = 1;  // 1 -> sub-buffer is locked by (this) wave
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

    // Have the calling wave release its exclusive read/write access to a sub-buffer
    __device__ inline void wave_release(uint8_t *atomic_flags, size_t sub_buf_idx, unsigned int active_lane_id)
    {
        if (active_lane_id == 0) // only one lane releases the sub-buffer on behalf of the wave
        {
            uint8_t flag_value = 0; // 0 -> sub-buffer is not locked
            __atomic_store(&atomic_flags[sub_buf_idx], &flag_value, __ATOMIC_RELEASE);
        }
    }

    // Have the calling wave, which has exclusive read/write access to a sub-buffer, pass its lock
    // on the sub-buffer to the host, such that the host can process and clear the sub-buffer.
    // The argument return_control tells host code whether is should return control back to the calling
    // wave, or if the sub-buffer should be released so that any wave can acquire it.
    __device__ inline void wave_signal_host(uint8_t *atomic_flags, size_t sub_buf_idx, unsigned int active_lane_id, bool return_control)
    {
        if (active_lane_id == 0) // only one lane passes control of the sub-buffer to the host on behalf of the wave
        {
            uint8_t flag_value = return_control ? 3 : 2;
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
    // End of section to be skipped by Doxygen
    //! @endcond

    //! \brief Submit a message of any type that is valid in device code from the device to the host.
    //!
    //! Messages are submitted on a per-wave basis, and only the active lanes in the wave submit.
    template <typename message_t>
    __device__ inline void v_submit_message(dh_comms_resources *rsrc, //!< Pointer to dh_comms device resources used for message submission.
                                                                      //!< This pointer is acquired by host code by calling dh_comms::get_dev_rsrc_ptr(),
                                                                      //!< and passed as a kernel argument to kernels that want to use v_submit_message().
                                            const message_t &message, //!< Message to be submitted.
                                            uint32_t user_type)       //!< Tag to distinguish between different kinds of messages, used by host
                                                                      //!< code that processes the messages.
    {
        unsigned int active_lane_id = __active_lane_id();               // One lane is doing some operations on behalf of the whole wave. That must be
                                                                        // a lane that is active, i.e., not masked out in the execution mask.
                                                                        // Also: all active lanes write data, and they use the active lane id to determine
                                                                        // the offset in the buffer for writing the lane data.
        size_t sub_buf_idx = get_sub_buffer_idx(rsrc->no_sub_buffers_); // Index of the sub-buffer the wave is going to use
        uint64_t exec = __builtin_amdgcn_read_exec();                   // Execution mask is passed in the wave header; can be used for analysis purposes.
        unsigned int active_lane_count = __active_lane_count(exec);     // Passed in the wave header, and used to compute the total size of the message for all lanes.
        size_t *sub_buffer_sizes = rsrc->sub_buffer_sizes_;             // Size of the data currently in our sub-buffer, in bytes
        size_t sub_buffer_capacity = rsrc->sub_buffer_capacity_;        // Max number of bytes thatfit into the sub-buffer
        char *buffer = rsrc->buffer_;                                   // Pointer to the main data buffer, of which our sub-buffer is a slice.

                                                                        // size of the data after the wave header. Data is written in multiples of
                                                                        // four bytes, so round up to multiple of four.
        uint64_t data_size = active_lane_count * (sizeof(lane_header_t) + 4 * ((sizeof(message_t) + 3) / 4));

        wave_acquire(rsrc->atomic_flags_, sub_buf_idx, active_lane_id); // Get exclusive access to the sub-buffer and associated data
        if (sizeof(wave_header_t) + data_size > sub_buffer_capacity)    // Sanity check: does the message even fit in an empty sub-buffer?
        {
            *rsrc->error_bits_ |= 1;                                    // If not, set an error bit, don't write any data, and return
            wave_release(rsrc->atomic_flags_, sub_buf_idx, active_lane_id);
            return;
        }
        size_t current_size = sub_buffer_sizes[sub_buf_idx];            // Does the message fit in the sub-buffer, given the data already in the sub-buffer?
        if (current_size + sizeof(wave_header_t) + data_size > sub_buffer_capacity)
        {                                                               // If not, tell the host to clear the sub-buffer, and to return control to us when it's done.
            bool return_control = true;
            wave_signal_host(rsrc->atomic_flags_, sub_buf_idx, active_lane_id, return_control);
            // The host clears the sub-buffer and resets its size to 0. Since it
            // returns control to us, no other wave can have changed the sub -buffer
            // size after wave_signal_host returns. So instead of re-reading the size
            // from memory, we can deduce that it must be 0. This saves a slow memory
            // read.
            current_size = 0;
        }
        // First write the wave header; only one wave takes care of that
        if (active_lane_id == 0)
        {
            wave_header_t wave_header(exec, data_size, active_lane_count, user_type);
            size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size;
            wave_header_t *wave_header_p = (wave_header_t *)(&buffer[byte_offset]);
            *wave_header_p = wave_header;
        }

        // Write the lane headers for the active lanes
        // Only active lanes write, and they write their headers
        size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size + sizeof(wave_header_t) + active_lane_id * sizeof(lane_header_t);
        lane_header_t *lane_header_p = (lane_header_t *)(&buffer[byte_offset]);
        lane_header_t lane_header;
        *lane_header_p = lane_header;

        // Write the message one dword at a time in a coalesced fashion
        // All active lanes should start writing after the lane headers, with the offsets
        // of consecutive active lanes 4 bytes apart.
        if constexpr (sizeof(lane_header_t) == sizeof(uint32_t))
        {
            // In the initial/current implementation, lane headers are 4 bytes,
            // which makes updating the offset simple and efficient, since
            // the offsets for consecutive active lanes are already four bytes apart.
            byte_offset += active_lane_count * sizeof(lane_header_t);
        }
        else
        {
            // If we ever add data to the lane headers so that the size changes.
            // offsets for consecutive lanes are no longer four bytes apart after
            // writing the lane headers, so we need a more complicated/expensive
            // calculation.
            byte_offset =
                sub_buf_idx * sub_buffer_capacity            // start of our sub-buffer
                + current_size                               // skipping past data that was already there
                + sizeof(wave_header_t)                      // skipping past the single wave header
                + active_lane_count * sizeof(lane_header_t)  // skipping past the lane headers
                + active_lane_id * sizeof(uint32_t);         // four bytes between consecutive active lanes
        }

        // Write the message four bytes at a time, taking care of
        // the tail of the message, which may have fewer than
        // four byes.
        size_t remaining_bytes = sizeof(message);
        char *src = (char *)&message;
        char *dst = &(buffer)[byte_offset];
        while (remaining_bytes)
        {
            size_t size = min(remaining_bytes, sizeof(uint32_t));
            memcpy(dst, src, size);
            src += sizeof(uint32_t);
            dst += active_lane_count * sizeof(uint32_t);
            remaining_bytes -= size;
        }

        // Update sub-buffer size and release lock
        if (active_lane_id == 0) // Only one wave must update the sub-buffer size
        {
            sub_buffer_sizes[sub_buf_idx] += sizeof(wave_header_t) + data_size;
        }
        wave_release(rsrc->atomic_flags_, sub_buf_idx, active_lane_id);
    }

} // namespace dh_comms