#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

#include <hip/hip_runtime.h>
#include "hip_utils/hip_utils.h"

#include "buffer.h"
#include "packet.h"

namespace
{
    constexpr size_t max_xcc = 16;       // 4 bits
    constexpr size_t max_se_per_xcc = 8; // 3 bits
    constexpr size_t max_cu_per_se = 16; // 4 bits
    constexpr size_t max_cu_per_device = max_xcc * max_se_per_xcc * max_cu_per_se;
    constexpr uint16_t all_ones = 0xffff;

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
     * This function takes the cu id, and prints the XCC, SE and CU
     * values, separated by colons.
     */
    void print_cu_id(uint16_t cu_id)
    {
        uint16_t xcc = (cu_id & 0x780) >> 7;
        uint16_t se = (cu_id & 0x70) >> 4;
        uint16_t cu = (cu_id & 0xf);
        printf("%02u:%02u:%02u", xcc, se, cu);
    }


    __global__ void report_cu_ids(uint16_t *id)
    {
        if (threadIdx.x == 0)
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
            uint16_t cu_id = xcc | se | cu;
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

    std::vector<uint16_t> invert_map(const std::vector<uint16_t> &index_to_cu_map)
    {
        std::vector<uint16_t> cu_to_index_map(max_cu_per_device, all_ones);
        for (int i = 0; i != index_to_cu_map.size(); ++i)
        {
            cu_to_index_map[index_to_cu_map[i]] = i;
        }

        return cu_to_index_map;
    }

    void *allocate_buffer(size_t no_sub_buffers, size_t packets_per_sub_buffer)
    {
        void *buffer;
        std::size_t size = no_sub_buffers * packets_per_sub_buffer * dh_comms::bytes_per_packet;
        printf("Allocating %lu bytes for %lu sub-buffers\n", size, no_sub_buffers);
        CHK_HIP_ERR(hipMalloc(&buffer, size));
        return buffer;
    }

} // unnamed namespace

namespace dh_comms
{

    buffer::buffer(std::size_t packets_per_sub_buffer)
        : no_sub_buffers_(get_cu_count()),
          index_to_cu_map_(determine_index_to_cu_map()),
          cu_to_index_map_(invert_map(index_to_cu_map_)),
          buffer_(allocate_buffer(no_sub_buffers_, packets_per_sub_buffer))
    {
        // Body intentionally empty
    }

    buffer::~buffer()
    {
        CHK_HIP_ERR(hipFree(buffer_));
    }

    void buffer::print_cu_to_index_map() const
    {
        printf("xcc:se:cu\n");
        for (int i = 0; i != cu_to_index_map_.size(); ++i)
        {
            // The index of non-exisiting/non-enabled CU is set to 0xffff.
            // We don't print those.
            if (cu_to_index_map_[i] != 0xffff)
            {
                print_cu_id(i);
                printf(" index = %u\n", cu_to_index_map_[i]);
            }
        }
    }

} // namespace dh_comms