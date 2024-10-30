#pragma once

#include "data_headers.h"
#include "hip_utils.h"

namespace dh_comms
{
    __device__ inline wave_header_t::wave_header_t(uint64_t exec, uint64_t data_size, bool is_vector_message, bool has_lane_headers,
                                                   uint64_t timestamp,
                                                   uint32_t active_lane_count, uint32_t src_loc_idx,
                                                   uint32_t user_type, uint32_t user_data)
        : exec(exec),
          data_size(data_size),
          is_vector_message(is_vector_message),
          has_lane_headers(has_lane_headers),
          timestamp(timestamp),
          active_lane_count(active_lane_count),
          src_loc_idx(src_loc_idx),
          user_type(user_type),
          user_data(user_data),
          block_idx_x(blockIdx.x),
          block_idx_y(blockIdx.y),
          block_idx_z(blockIdx.z),
          wave_num(__wave_num())
    {
        // for pre-MI300 hardware that isn't partitioned into XCCx, we set the xcc_id to zero. For
        // the MI300 variants (gfx94[012], we query the hardware registers to find out where on the
        // device we are running. Documentation of the hardware registers can be found in Section
        // 3.12 of the AMD Instinct MI300 Instruction Set Architecture Reference Guide at
        // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf.
        // Note that a wave may be preempted from the XCC|SE|CU on which it started and resumed on
        // another XCC|SE|CU, so it's not safe to rely on these values.
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

    __device__ inline lane_header_t::lane_header_t()
    {
        thread_idx_x = threadIdx.x;
        thread_idx_y = threadIdx.y;
        thread_idx_z = threadIdx.z;
    }

} // namespace dh_comms