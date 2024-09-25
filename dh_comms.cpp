#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

#include <hip/hip_runtime.h>
#include "hip_utils.h"

#include "dh_comms.h"
#include "data_headers.h"

namespace
{
    constexpr bool shared_buffers_are_host_pinned = true;

    void *allocate_shared_buffer(size_t size)
    {
        void *buffer;
        std::vector<char> zeros(size);
        if constexpr (shared_buffers_are_host_pinned)
        {
            CHK_HIP_ERR(hipHostMalloc(&buffer, size, hipHostMallocCoherent));
            std::copy(zeros.cbegin(), zeros.cend(), (char *)buffer);
        }
        else
        {
            CHK_HIP_ERR(hipExtMallocWithFlags(&buffer, size, hipDeviceMallocFinegrained));
            CHK_HIP_ERR(hipMemcpy(buffer, zeros.data(), size, hipMemcpyHostToDevice));
        }
        return buffer;
    }

    template <typename T>
    T *clone_to_device(const std::vector<T> &host_vec)
    {
        T *buffer;
        size_t size = host_vec.size() * sizeof(T);
        CHK_HIP_ERR(hipMalloc(&buffer, size));
        CHK_HIP_ERR(hipMemcpy(buffer, host_vec.data(), size, hipMemcpyHostToDevice));
        return buffer;
    }

} // unnamed namespace

namespace dh_comms
{
    __constant__ size_t no_sub_buffers_c;
    __constant__ size_t sub_buffer_capacity_c;
    __constant__ char *buffer_c;
    __constant__ size_t *sub_buffer_sizes_c;
    __constant__ uint8_t *atomic_flags_c;

    dh_comms::dh_comms(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity,
                   message_processor_base& message_processor,
                   std::size_t no_host_threads)
        : no_sub_buffers_(no_sub_buffers),
          sub_buffer_capacity_(sub_buffer_capacity),
          message_processor_(message_processor),
          buffer_((decltype(buffer_))allocate_shared_buffer(no_sub_buffers_ * sub_buffer_capacity_)),
          sub_buffer_sizes_((decltype(sub_buffer_sizes_))allocate_shared_buffer(no_sub_buffers_ * sizeof(decltype(*sub_buffer_sizes_)))),
          atomic_flags_((decltype(atomic_flags_))allocate_shared_buffer(no_sub_buffers_ * sizeof(decltype(*atomic_flags_)))),
          teardown_(false),
          sub_buffer_processors_(init_host_threads(no_host_threads))
    {
        if constexpr (shared_buffers_are_host_pinned)
        {
            printf("%s:%d:\n\t Buffers accessed from both host and device are allocated in pinned host memory\n",
                   __FILE__, __LINE__);
        }
        else
        {
            printf("%s:%d:\n\t Buffers accessed from both host and device are allocated in device memory\n",
                   __FILE__, __LINE__);
        }

        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(no_sub_buffers_c),
                                      &no_sub_buffers_, sizeof(no_sub_buffers_)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(sub_buffer_capacity_c),
                                      &sub_buffer_capacity_, sizeof(sub_buffer_capacity_)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(buffer_c),
                                      &buffer_, sizeof(void *)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(sub_buffer_sizes_c),
                                      &sub_buffer_sizes_, sizeof(void *)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(atomic_flags_c),
                                      &atomic_flags_, sizeof(void *)));
    }

    dh_comms::~dh_comms()
    {
        CHK_HIP_ERR(hipDeviceSynchronize());
        teardown_ = true;
        for (auto &sbp : sub_buffer_processors_)
        {
            sbp.join();
        }

        CHK_HIP_ERR(hipFree(sub_buffer_sizes_));
        CHK_HIP_ERR(hipFree(atomic_flags_));
        CHK_HIP_ERR(hipFree(buffer_));
    }

    std::vector<std::thread> dh_comms::init_host_threads(std::size_t no_host_threads)
    {
        assert(no_host_threads != 0);
        std::size_t no_sub_buffers_per_thread = no_sub_buffers_ / no_host_threads;
        std::size_t remainder = no_sub_buffers_ % no_host_threads;

        std::vector<std::thread> sub_buffer_processors;
        std::size_t first = 0;
        std::size_t last;
        for (std::size_t i = 0; i != no_host_threads; ++i)
        {
            last = first + no_sub_buffers_per_thread;
            if (i < remainder)
            {
                ++last;
            }
            sub_buffer_processors.emplace_back(std::thread(&dh_comms::process_sub_buffers, this, first, last));
            first = last;
        }
        assert(last == no_sub_buffers_);

        return sub_buffer_processors;
    }

    void dh_comms::process_sub_buffers(std::size_t first, std::size_t last)
    {
        while (not teardown_)
        {
            for (size_t i = first; i != last; ++i)
            {
                // when the sub-buffer for a wave on the device is full, it will
                // set the flag to either 3 (if it wants control back after the host
                // is done processing the sub-buffer) or 2 (if it doesn't want control back,
                // and instead allows any wave to take control of the sub-buffer)
                uint8_t flag = __atomic_load_n(&atomic_flags_[i], __ATOMIC_ACQUIRE);
                if (flag > 1) // buffer is full: process and reset
                {
                    // TODO: process data
                    size_t size = sub_buffer_sizes_[i];
                    size_t byte_offset = i * sub_buffer_capacity_;
                    char *message_p = &buffer_[byte_offset];
                    while (size != 0)
                    {
                        size = message_processor_(message_p, size, i);
                    }

                    sub_buffer_sizes_[i] = 0;
                    // setting the flag to either 1 (giving contol back to the signaling wave)
                    // or 0 (allowing any wave to take control of the sub-buffer)
                    __atomic_store_n(&atomic_flags_[i], flag - 2, __ATOMIC_RELEASE);
                }
            }
        }

        // printf("[Host] process_sub_buffers: processing partially full sub-buffers after kernels have finished\n");

        for (size_t i = first; i != last; ++i)
        {
            uint8_t flag = __atomic_load_n(&atomic_flags_[i], __ATOMIC_ACQUIRE);
            if (flag != 0) // Should not happen, indicates a missing atomic release from device code
            {
                printf("Found non-zero flag for sub-buffer %lu\n", i);
            }
            // TODO: process data
            size_t size = sub_buffer_sizes_[i];
            size_t byte_offset = i * sub_buffer_capacity_;
            char *message_p = &buffer_[byte_offset];
            while (size != 0)
            {
                size = message_processor_(message_p, size, i);
            }
        }
    }

} // namespace dh_comms

// device functions
namespace dh_comms
{
    __device__ size_t no_sub_buffers_f()
    {
        return no_sub_buffers_c;
    }

    __device__ size_t sub_buffer_capacity_f()
    {
        return sub_buffer_capacity_c;
    }

    __device__ char *buffer_f()
    {
        return buffer_c;
    }

    __device__ size_t *sub_buffer_sizes_f()
    {
        return sub_buffer_sizes_c;
    }

    __device__ uint8_t *atomic_flags_f()
    {
        return atomic_flags_c;
    }

    __device__ size_t get_sub_buffer_idx()
    {
        size_t grid_dim_x_m = gridDim.x % no_sub_buffers_c;
        size_t grid_dim_y_m = gridDim.y % no_sub_buffers_c;
        size_t grid_dim_xy_m = (grid_dim_x_m * grid_dim_y_m) % no_sub_buffers_c;
        size_t block_id_z_grid_dim_xy_m = (blockIdx.z * grid_dim_xy_m) % no_sub_buffers_c;

        size_t block_id_y_grid_dim_x_m = (blockIdx.y * grid_dim_x_m) % no_sub_buffers_c;

        size_t block_id_x_m = blockIdx.x % no_sub_buffers_c;

        return (block_id_z_grid_dim_xy_m + block_id_y_grid_dim_x_m + block_id_x_m) % no_sub_buffers_c;
    }

    __device__ void wave_acquire(size_t sub_buf_idx, unsigned int active_lane_id)
    {
        if (active_lane_id == 0)
        {
            uint8_t expected = 0;
            uint8_t desired = 1;
            bool weak = false;
            while (not __atomic_compare_exchange(&atomic_flags_c[sub_buf_idx], &expected, &desired, weak,
                                                 __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            {
                // __atomic_compare_exchange returns the value at the address (first argument) in
                // the second argument (expected), so we need to reset it in this spin-loop
                expected = 0;
            }
        }
    }

    __device__ void wave_release(size_t sub_buf_idx, unsigned int active_lane_id)
    {
        if (active_lane_id == 0)
        {
            uint8_t flag_value = 0;
            __atomic_store(&atomic_flags_c[sub_buf_idx], &flag_value, __ATOMIC_RELEASE);
        }
    }

    __device__ void wave_signal_host(size_t sub_buf_idx, unsigned int active_lane_id, bool return_control)
    {
        if (active_lane_id == 0)
        {
            uint8_t flag_value = return_control ? 3 : 2;
            // printf("[Device] signalling buffer full for idx %lu\n", sub_buf_idx);
            __atomic_store(&atomic_flags_c[sub_buf_idx], &flag_value, __ATOMIC_RELEASE);
            // when host code sees a flag value > 1, it starts processing the sub-buffer that
            // we indicated to be full. Once the host is done, it subtracts 2 from the flag,
            // i.e., if return_control is true, the flag will be 1 after the host is done,
            // and this wave will wait for that to take control. Otherwise, if return_control
            // is false, the host will set the flag to 0, and any wave that wants to write
            // to the sub-buffer can get control by calling wave_acquire()
            if (return_control)
            {
                // spin until host sets flag to 1, indicating it returns control to this wave
                while (__atomic_load_n(&atomic_flags_c[sub_buf_idx], __ATOMIC_ACQUIRE) != 1)
                {
                }
            }
        }
    }

}