#pragma once

#include <cstdio>
#include <hip/hip_runtime.h>

inline int checkHipError(const hipError_t err, const char *cmd) {
  if (err) {
    printf("HIP error in command '%s'\n", cmd);
    printf("Error message: %s\n", hipGetErrorString(err));
  }
  return err;
}

#define CHK_HIP_ERR(cmd) checkHipError(cmd, #cmd)

// HIP already defines a __lane_id() device function, which gives the lane number for
// any thread in a possibly higher-dimensional thread block

// similar to __lane_id(), but only consider active lanes. E.g., if lanes
// 0 and 1 are masked out in the execution mask, and the remaining 62 lanes
// are active, then lanes 2..63 would have active lane ids 0..61, respectively
__device__ static inline unsigned int __active_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(__builtin_amdgcn_read_exec_hi(),
                                   __builtin_amdgcn_mbcnt_lo(__builtin_amdgcn_read_exec_lo(), 0u));
}

// returns the number of active lanes, based on the execution mask passed as argument.
__device__ static inline unsigned int __active_lane_count(uint64_t exec) {
  unsigned int active_lanes_count;
  asm volatile("s_bcnt1_i32_b64 %0 %1" : "=s"(active_lanes_count) : "s"(exec));
  return active_lanes_count;
}

// returns the number of active lanes, based on the current execution mask.
__device__ static inline unsigned int __active_lane_count() {
  uint64_t exec = __builtin_amdgcn_read_exec();
  return __active_lane_count(exec);
}

// HIP provides __builtin_amdgcb_read_exec to read the execution mask, but
// no function to set the execution mask. This function provides the latter
__device__ static inline void __set_exec(uint64_t exec) { asm volatile("s_mov_b64 exec, %0" : : "s"(exec)); }

// Returns the wave number in the thread block for any thread, based on its possibly
// higher-dimensional thread coordinates.
__device__ static inline unsigned int __wave_num() {
  return (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) / 64;
}
