#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

#include <hip/hip_runtime.h>
#include "hip_utils.h"

#include "buffer.h"
#include "packet.h"

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
    __constant__ size_t packets_per_sub_buffer_c;
    __constant__ void *packet_buffer_c;
    __constant__ size_t *sub_buffer_sizes_c;
    __constant__ uint8_t *atomic_flags_c;

    buffer::buffer(std::size_t no_sub_buffers, std::size_t packets_per_sub_buffer, std::size_t no_host_threads)
        : no_sub_buffers_(no_sub_buffers),
          packets_per_sub_buffer_(packets_per_sub_buffer),
          packet_buffer_(allocate_shared_buffer(no_sub_buffers_ * packets_per_sub_buffer_ * bytes_per_packet)),
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
        printf("from buffer ctor:\n");
        printf("no_sub_buffers_ = %lu\n", no_sub_buffers_);
        printf("packets_per_sub_buffer_ = %lu\n", packets_per_sub_buffer_);

        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(no_sub_buffers_c),
                                      &no_sub_buffers_, sizeof(no_sub_buffers_)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(packets_per_sub_buffer_c),
                                      &packets_per_sub_buffer_, sizeof(packets_per_sub_buffer_)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(packet_buffer_c),
                                      &packet_buffer_, sizeof(void *)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(sub_buffer_sizes_c),
                                      &sub_buffer_sizes_, sizeof(void *)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(atomic_flags_c),
                                      &atomic_flags_, sizeof(void *)));
    }

    buffer::~buffer()
    {
        CHK_HIP_ERR(hipDeviceSynchronize());
        teardown_ = true;
        for (auto &sbp : sub_buffer_processors_)
        {
            sbp.join();
        }

        CHK_HIP_ERR(hipFree(sub_buffer_sizes_));
        CHK_HIP_ERR(hipFree(atomic_flags_));
        CHK_HIP_ERR(hipFree(packet_buffer_));
    }

    std::vector<std::thread> buffer::init_host_threads(std::size_t no_host_threads)
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
            sub_buffer_processors.emplace_back(std::thread(&buffer::process_sub_buffers, this, first, last));
            first = last;
        }
        assert(last == no_sub_buffers_);

        return sub_buffer_processors;
    }

    void buffer::show_queues() const
    {
        for (size_t i = 0; i != no_sub_buffers_; ++i)
        {
            size_t size = sub_buffer_sizes_[i];
            if (size != 0)
            {
                printf("[Host] sub_buffer %lu: size = %lu\n", i, size);
                packet *packets = (packet *)packet_buffer_;
                packets = &packets[i * packets_per_sub_buffer_];
                for (size_t i = 0; i != size; ++i)
                {
                    printf("[Host]  %s\n", packet_str(packets[i]).c_str());
                }
            }
        }
    }

    void buffer::process_sub_buffers(std::size_t first, std::size_t last)
    {
        std::size_t no_packets = 0;
        std::size_t sum = 0;
        while (not teardown_)
        {
            for (size_t i = first; i != last; ++i)
            {
                // uint8_t expected = 2;
                // uint8_t desired = 3;
                // bool weak = false;
                // if(__atomic_compare_exchange(&atomic_flags_[i], &expected, &desired, weak, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
                uint8_t flag = __atomic_load_n(&atomic_flags_[i], __ATOMIC_ACQUIRE);
                if (flag == 2) // buffer is full: process and reset
                {
                    // TODO: process data
                    no_packets += sub_buffer_sizes_[i];
                    packet *packets = (packet *)packet_buffer_;
                    packets = &packets[i * packets_per_sub_buffer_];
                    for (size_t p = 0; p != sub_buffer_sizes_[i]; ++p)
                    {
                        sum += packets[p].wg_x;
                    }
                    // printf("[Host] process_sub_buffers: sub-buffer %lu has size %lu\n", i, sub_buffer_sizes_[i]);
                    // After processing: set size to zero, and reset flag
                    sub_buffer_sizes_[i] = 0;
                    __atomic_store_n(&atomic_flags_[i], 0, __ATOMIC_RELEASE);
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
            no_packets += sub_buffer_sizes_[i];
            packet *packets = (packet *)packet_buffer_;
            packets = &packets[i * packets_per_sub_buffer_];
            for (size_t p = 0; p != sub_buffer_sizes_[i]; ++p)
            {
                sum += packets[p].wg_x;
            }
            // printf("[Host] process_sub_buffers: sub-buffer %lu has size %lu\n", i, sub_buffer_sizes_[i]);
        }
        printf("[Host] processed %lu packets, sum of wg_x = %lu\n", no_packets, sum);
    }

} // namespace dh_comms

// device functions
namespace dh_comms
{
    __device__ uint16_t get_cu_id()
    {
        uint16_t xcc = 0;
#if defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
        uint32_t xcc_reg;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_reg));
        xcc = (xcc_reg & 0xf) << 7;
#endif
        uint32_t cu_se_reg;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(cu_se_reg));
        uint16_t se = ((cu_se_reg >> 13) & 0x7) << 4;
        uint16_t cu = (cu_se_reg >> 8) & 0xf;
        return xcc | se | cu;
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

    __device__ void wave_acquire(size_t sub_buf_idx)
    {
        // TODO: below test may not match first thread of wave, depending on block dimensions
        // TODO: we probably need to store the exec mask, set it to 1, and restore afterwards
        if (threadIdx.x % 64 == 0)
        {
            uint8_t expected = 0;
            uint8_t desired = 1;
            bool weak = false;
            // printf("aqcuiring flag for idx %lu: start\n", sub_buf_idx);
            while (not __atomic_compare_exchange(&atomic_flags_c[sub_buf_idx], &expected, &desired, weak,
                                                 __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            {
                // printf("aqcuiring flag for idx %lu: spinning\n", sub_buf_idx);
                expected = 0;
            }
            // printf("aqcuiring flag for idx %lu: success\n", sub_buf_idx);
        }
    }

    __device__ void wave_release(size_t sub_buf_idx)
    {
        // TODO: below test may not match first thread of wave, depending on block dimensions
        // TODO: we probably need to store the exec mask, set it to 1, and restore afterwards
        if (threadIdx.x % 64 == 0)
        {
            uint8_t zero = 0;
            // printf("releasing flag for idx %lu: start\n", sub_buf_idx);
            __atomic_store(&atomic_flags_c[sub_buf_idx], &zero, __ATOMIC_RELEASE);
            // printf("releasing flag for idx %lu: done\n", sub_buf_idx);
        }
    }

    __device__ void wave_signal_host(size_t sub_buf_idx)
    {
        // TODO: below test may not match first thread of wave, depending on block dimensions
        // TODO: we probably need to store the exec mask, set it to 1, and restore afterwards
        if (threadIdx.x % 64 == 0)
        {
            uint8_t two = 2;
            // printf("[Device] signalling buffer full for idx %lu\n", sub_buf_idx);
            __atomic_store(&atomic_flags_c[sub_buf_idx], &two, __ATOMIC_RELEASE);
        }
    }

    __device__ void submit_packet(const packet &p)
    {
        bool can_write = false;
        size_t sub_buf_idx = get_sub_buffer_idx();
        while (not can_write)
        {
            wave_acquire(sub_buf_idx);
            if (sub_buffer_sizes_c[sub_buf_idx] + 64 > packets_per_sub_buffer_c)
            {
                // insufficient space for all threads in the wave to write ->
                // signal to the host that the buffer is full so that it can process and
                // purge the buffer
                wave_signal_host(sub_buf_idx);
            }
            else
            {
                can_write = true;
            }
        }
        packet *packet_buffer = (packet *)packet_buffer_c;
        packet_buffer = &packet_buffer[sub_buf_idx * packets_per_sub_buffer_c + sub_buffer_sizes_c[sub_buf_idx] + threadIdx.x];
        *packet_buffer = p;
        // TODO: size increment needs to be done on a per-wave basis with a scalar instruction
        atomicAdd(&sub_buffer_sizes_c[sub_buf_idx], 1);
        wave_release(sub_buf_idx);
    }
}