// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "dh_comms.h"
#include "dh_comms_dev.h"
#include "hip_utils.h"
#include "memory_analysis_handler.h"

#include <cstring>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <string>

__global__ void test_uint32_t(int *indices, uint32_t *sink, dh_comms::dh_comms_descriptor *rsrc) {
  __shared__ uint32_t lds[64 * 1024 / sizeof(uint32_t)];
  lds[threadIdx.x] = threadIdx.x;
  dh_comms::v_submit_address(rsrc, lds + threadIdx.x, 0, __LINE__, 0, dh_comms::memory_access::write,
                             dh_comms::address_space::shared, sizeof(uint32_t));

  int idx = indices[threadIdx.x];
  if (idx == -1) {
    return;
  }
  uint32_t garbage = lds[idx];
  dh_comms::v_submit_address(rsrc, lds + idx, 0, __LINE__, 0, dh_comms::memory_access::read,
                             dh_comms::address_space::shared, sizeof(uint32_t));
  garbage += lds[idx + 64];
  dh_comms::v_submit_address(rsrc, lds + idx + 64, 0, __LINE__, 0, dh_comms::memory_access::read,
                             dh_comms::address_space::shared, sizeof(uint32_t));
  *sink = garbage;
}

void set_indices(int *indices, const std::vector<std::pair<int, int>> &kvs, bool clear = false) {
  if (clear) {
    for (int i = 0; i != 64; ++i) {
      indices[i] = -1;
    }
  }
  for (auto kv : kvs) {
    assert(kv.first >= -1 and kv.first < 64);
    indices[kv.first] = kv.second;
  }
}

int main() {
  // set default parameter values
  //    kernel launch parameters
  constexpr int blocksize = 64;
  constexpr int no_blocks = 1;

  //    dh_comms configuration parameters
  size_t no_sub_buffers = 256; // gave best performance in several not too thorough tests
  size_t sub_buffer_capacity = 64 * 1024;
  bool verbose = true; // only be verbose with really small array sizes

  int *indices;
  CHK_HIP_ERR(hipHostMalloc(&indices, 64 * sizeof(int), hipHostMallocNonCoherent));
  for (std::size_t i = 0; i != 64; ++i) {
    indices[i] = -1;
  }
  void *sink;
  CHK_HIP_ERR(hipMalloc(&sink, 32));

  {
    dh_comms::dh_comms dh_comms(no_sub_buffers, sub_buffer_capacity, verbose);
    dh_comms.append_handler(std::make_unique<dh_comms::memory_analysis_handler_t>(verbose));
    dh_comms.start();
    // if dh_comms sub-buffers get full during running of the kernel,
    // device code notifies host code to process the full buffers and
    // clear them
    set_indices(indices, {{0, 0}, {1, 32}, {8, 64}}, true);
    test_uint32_t<<<no_blocks, blocksize>>>(indices, (uint32_t *)sink, dh_comms.get_dev_rsrc_ptr());
    CHK_HIP_ERR(hipDeviceSynchronize());
    set_indices(indices, {{0, 96}, {1, 128}, {8, 160}}, true);
    test_uint32_t<<<no_blocks, blocksize>>>(indices, (uint32_t *)sink, dh_comms.get_dev_rsrc_ptr());
    // make sure kernels are done before stopping dh_comms, or device messages will get lost
    CHK_HIP_ERR(hipDeviceSynchronize());
    dh_comms.stop();
    dh_comms.report();

    dh_comms.start();
    set_indices(indices, {{12, 192}, {13, 224}, {32, 192}, {33, 224}, {34, 0}}, true);
    test_uint32_t<<<no_blocks, blocksize>>>(indices, (uint32_t *)sink, dh_comms.get_dev_rsrc_ptr());
    CHK_HIP_ERR(hipDeviceSynchronize());
    dh_comms.stop();
    dh_comms.report();
  }

  CHK_HIP_ERR(hipFree(sink));
  CHK_HIP_ERR(hipFree(indices));
}
