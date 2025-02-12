#include "dh_comms.h"
#include "dh_comms_dev.h"
#include "hip_utils.h"
#include "memory_analysis_handler.h"

#include <cstring>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <string>

__global__ void test_uint64_t(int *indices, uint64_t *memory, dh_comms::dh_comms_descriptor *rsrc) {
  int idx = indices[threadIdx.x];
  dh_comms::v_submit_address(rsrc, indices + threadIdx.x, 0, __LINE__, 0, dh_comms::memory_access::read,
                             dh_comms::address_space::global, sizeof(int));
  if (idx == -1) {
    return;
  }
  memory[idx] = 0;
  dh_comms::v_submit_address(rsrc, memory + idx, 0, __LINE__, 0, dh_comms::memory_access::write,
                             dh_comms::address_space::global, sizeof(uint64_t));
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
  void *memory;
  CHK_HIP_ERR(hipMalloc(&memory, 64 * 1024));

  {
    dh_comms::dh_comms dh_comms(no_sub_buffers, sub_buffer_capacity, verbose);
    dh_comms.append_handler(std::make_unique<dh_comms::memory_analysis_handler_t>(verbose));
    dh_comms.start();
    // if dh_comms sub-buffers get full during running of the kernel,
    // device code notifies host code to process the full buffers and
    // clear them
    set_indices(indices, {{0, 0}, {1, 32}, {8, 64}}, true);
    test_uint64_t<<<no_blocks, blocksize>>>(indices, (uint64_t *)memory, dh_comms.get_dev_rsrc_ptr());
    CHK_HIP_ERR(hipDeviceSynchronize());
    set_indices(indices, {{0, 96}, {1, 128}, {8, 160}}, true);
    test_uint64_t<<<no_blocks, blocksize>>>(indices, (uint64_t *)memory, dh_comms.get_dev_rsrc_ptr());
    // make sure kernels are done before stopping dh_comms, or device messages will get lost
    CHK_HIP_ERR(hipDeviceSynchronize());
    dh_comms.stop();
    dh_comms.report();

    dh_comms.start();
    set_indices(indices, {{12, 192}, {13, 224}, {32, 192}, {33, 224}, {34, 0}}, true);
    test_uint64_t<<<no_blocks, blocksize>>>(indices, (uint64_t *)memory, dh_comms.get_dev_rsrc_ptr());
    CHK_HIP_ERR(hipDeviceSynchronize());
    dh_comms.stop();
    dh_comms.report();
  }

  CHK_HIP_ERR(hipFree(memory));
  CHK_HIP_ERR(hipFree(indices));
}
