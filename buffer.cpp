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
    constexpr size_t max_xcc = 16;       // 4 bits
    constexpr size_t max_se_per_xcc = 8; // 3 bits
    constexpr size_t max_cu_per_se = 16; // 4 bits
    constexpr size_t max_cu_per_device = max_xcc * max_se_per_xcc * max_cu_per_se;
    constexpr uint16_t all_ones = 0xffff;
    constexpr bool shared_buffers_are_host_pinned = true;

    /***
     * Returns the number of compute units of the active HIP device
     */
    size_t get_cu_count()
    {
        hipDeviceProp_t props;
        CHK_HIP_ERR(hipGetDeviceProperties(&props, 0)); // TODO: handle multiple devices
        return props.multiProcessorCount;
    }

    /***
     * A CU id is an 11-bit value comprised of 4 bits for the XCC,
     * 3 bits for the SE on the XCC, and 4 bits for the CU on the SE.
     * This function takes the cu id, and returns a string with the XCC,
     * SE and CU values, separated by colons.
     */
    std::string cu_id_str(uint16_t cu_id)
    {
        char buffer[9] = {0};
        uint16_t xcc = (cu_id & 0x780) >> 7;
        uint16_t se = (cu_id & 0x70) >> 4;
        uint16_t cu = (cu_id & 0xf);
        sprintf(buffer, "%02u:%02u:%02u", xcc, se, cu);
        return std::string(buffer);
    }

    __global__ void report_cu_ids(uint16_t *id)
    {
        if (threadIdx.x == 0)
        {
            uint16_t cu_id = dh_comms::get_cu_id();
            id[cu_id] = cu_id;
        }
    }

    std::vector<uint16_t> determine_index_to_cu_map()
    {
        std::vector<uint16_t> cu_ids_h(max_cu_per_device, all_ones);
        uint16_t *cu_ids_d;

        CHK_HIP_ERR(hipMalloc(&cu_ids_d, max_cu_per_device * sizeof(uint16_t)));
        CHK_HIP_ERR(hipMemcpy(cu_ids_d, cu_ids_h.data(), max_cu_per_device * sizeof(uint16_t), hipMemcpyHostToDevice));
        report_cu_ids<<<10 * max_cu_per_device, 256>>>(cu_ids_d);
        CHK_HIP_ERR(hipMemcpy(cu_ids_h.data(), cu_ids_d, max_cu_per_device * sizeof(uint16_t), hipMemcpyDeviceToHost));
        CHK_HIP_ERR(hipFree(cu_ids_d));

        std::sort(cu_ids_h.begin(), cu_ids_h.end());
        auto last = std::unique(cu_ids_h.begin(), cu_ids_h.end());
        auto count = last - cu_ids_h.begin();
        if (count != 0 and cu_ids_h[count - 1] == all_ones)
        {
            --count;
        }
        cu_ids_h.resize(count);
        return cu_ids_h;
    }

    std::vector<size_t> invert_map(const std::vector<uint16_t> &index_to_cu_map)
    {
        std::vector<size_t> cu_to_index_map(max_cu_per_device, all_ones);
        for (size_t i = 0; i != index_to_cu_map.size(); ++i)
        {
            cu_to_index_map[index_to_cu_map[i]] = i;
        }

        return cu_to_index_map;
    }

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
    __constant__ size_t *cu_to_index_map_c;
    __constant__ uint8_t *atomic_flags_c;

    buffer::buffer(std::size_t packets_per_sub_buffer)
        : no_sub_buffers_(get_cu_count()),
          packets_per_sub_buffer_(packets_per_sub_buffer),
          index_to_cu_map_(determine_index_to_cu_map()),
          cu_to_index_map_(invert_map(index_to_cu_map_)),
          packet_buffer_(allocate_shared_buffer(no_sub_buffers_ * packets_per_sub_buffer_ * bytes_per_packet)),
          sub_buffer_sizes_((decltype(sub_buffer_sizes_))allocate_shared_buffer(no_sub_buffers_ * sizeof(decltype(*sub_buffer_sizes_)))),
          cu_to_index_map_d_(clone_to_device(cu_to_index_map_)),
          atomic_flags_((decltype(atomic_flags_))allocate_shared_buffer(no_sub_buffers_ * sizeof(decltype(*atomic_flags_)))),
          teardown_(false),
          buffer_processor_(std::thread(&buffer::process_buffer, this))
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
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(cu_to_index_map_c),
                                      &cu_to_index_map_d_, sizeof(void *)));
        CHK_HIP_ERR(hipMemcpyToSymbol(HIP_SYMBOL(atomic_flags_c),
                                      &atomic_flags_, sizeof(void *)));
    }

    buffer::~buffer()
    {
        CHK_HIP_ERR(hipDeviceSynchronize());
        teardown_ = true;
        buffer_processor_.join();

        CHK_HIP_ERR(hipFree(sub_buffer_sizes_));
        CHK_HIP_ERR(hipFree(atomic_flags_));
        CHK_HIP_ERR(hipFree(cu_to_index_map_d_));
        CHK_HIP_ERR(hipFree(packet_buffer_));
    }

    void buffer::print_cu_to_index_map() const
    {
        printf("xcc:se:cu\n");
        int column = 0;
        for (size_t i = 0; i != cu_to_index_map_.size(); ++i)
        {
            // The index of non-exisiting/non-enabled CU is set to 0xffff.
            // We don't print those.
            if (cu_to_index_map_[i] != 0xffff)
            {
                printf("%s index = %3lu", cu_id_str(i).c_str(), cu_to_index_map_[i]);
                if (++column % 4)
                {
                    printf("  |  ");
                }
                else
                {
                    printf("\n");
                }
            }
        }
    }

    void buffer::show_queues() const
    {
        for (size_t i = 0; i != no_sub_buffers_; ++i)
        {
            size_t size = sub_buffer_sizes_[i];
            if (size != 0)
            {
                printf("[Host] sub_buffer %lu for CU %s: size = %lu\n", i, cu_id_str(index_to_cu_map_[i]).c_str(), size);
                packet *packets = (packet *)packet_buffer_;
                packets = &packets[i * packets_per_sub_buffer_];
                for (size_t i = 0; i != size; ++i)
                {
                    printf("[Host]  %s\n", packet_str(packets[i]).c_str());
                }
            }
        }
    }

    void buffer::process_buffer()
    {
        std::size_t no_packets = 0;
        std::size_t sum = 0;
        while (not teardown_)
        {
            for (size_t i = 0; i != no_sub_buffers_; ++i)
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
                    // printf("[Host] process_buffer: sub-buffer %lu has size %lu\n", i, sub_buffer_sizes_[i]);
                    // After processing: set size to zero, and reset flag
                    sub_buffer_sizes_[i] = 0;
                    __atomic_store_n(&atomic_flags_[i], 0, __ATOMIC_RELEASE);
                }
            }
        }

        // printf("[Host] process_buffer: processing partially full sub-buffers after kernels have finished\n");

        for (size_t i = 0; i != no_sub_buffers_; ++i)
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
            // printf("[Host] process_buffer: sub-buffer %lu has size %lu\n", i, sub_buffer_sizes_[i]);
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

    __device__ size_t cu_to_index_map_f(uint16_t cu_id)
    {
        return cu_to_index_map_c[cu_id];
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
        uint16_t cu_id = get_cu_id();
        size_t sub_buf_idx = cu_to_index_map_f(cu_id);
        while (not can_write)
        {
            wave_acquire(sub_buf_idx);
            if (sub_buffer_sizes_c[sub_buf_idx] + blockDim.x > packets_per_sub_buffer_c)
            {
                // insufficient space for all threads in the block to write ->
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