#pragma once

#include "hip_utils.h"

namespace dh_comms
{

    struct wave_header_t
    {
        uint64_t exec;
        uint64_t data_size; // (number of active lanes) * (lane header (4 bytes) + number of data bytes per lane),
                            // rouded up to nearest multiple of 4 bytes
        uint32_t active_lane_count : 7;
        uint32_t unused32 : 25;
        uint32_t user_type;

        uint16_t block_idx_x;
        uint16_t block_idx_y;
        uint16_t block_idx_z;
        uint16_t wave_id : 4; // wave number withing workgroup; there can be at most 16 waves / workgroup
        uint16_t xcc_id : 4;  // zero for pre-MI300 hardware
        uint16_t se_id : 3;
        uint16_t cu_id : 4;
        uint16_t unused16 : 1;

        __device__ wave_header_t(uint64_t exec, uint64_t data_size, uint32_t active_lane_count, uint32_t user_type);
    };

    __device__ inline wave_header_t::wave_header_t(uint64_t exec, uint64_t data_size, uint32_t active_lane_count, uint32_t user_type)
        : exec(exec),
          data_size(data_size),
          active_lane_count(active_lane_count),
          user_type(user_type),
          block_idx_x(blockIdx.x),
          block_idx_y(blockIdx.y),
          block_idx_z(blockIdx.z),
          wave_id(__wave_id())
    {
        xcc_id = 0;
#if defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
        uint32_t xcc_reg;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_reg));
        xcc_id = (xcc_reg & 0xf);
#endif
        uint32_t cu_se_reg;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(cu_se_reg));
        se_id = ((cu_se_reg >> 13) & 0x7);
        cu_id = (cu_se_reg >> 8) & 0xf;
    }

    struct lane_header_t
    {
        uint32_t thread_idx_x : 10;
        uint32_t thread_idx_y : 10;
        uint32_t thread_idx_z : 10;
        uint32_t unused : 2;

        __device__ lane_header_t();
    };

    __device__ inline lane_header_t::lane_header_t()
    {
        thread_idx_x = threadIdx.x;
        thread_idx_y = threadIdx.y;
        thread_idx_z = threadIdx.z;
    }

} // namespace dh_comms