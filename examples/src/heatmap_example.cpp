#include <hip/hip_runtime.h>
#include "dh_comms.h"
#include "dh_comms_dev.h"
#include "hip_utils.h"
#include "memory_heatmap.h"

__global__ void test(float *dst, float *src, float alpha, size_t size, dh_comms::dh_comms_resources *rsrc)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    dst[idx] = alpha * src[idx];

    // since our buffers can have a mix of multiple types of messages,
    // we need to tag messages with a message type, so that
    // host processing code knows what to do with the message
    uint32_t user_defined_message_type = 0;
    dh_comms::v_submit_message(rsrc, dst + idx, user_defined_message_type);
    dh_comms::v_submit_message(rsrc, src + idx, user_defined_message_type);
}

int main()
{
    // kernel launch parameters
    constexpr size_t size = 5 * 1024 * 128 + 17;
    constexpr int threads_per_block = 128;
    constexpr int no_blocks = (size + threads_per_block - 1) / threads_per_block;
    constexpr size_t messages_per_wave = 2;
    constexpr size_t waves_per_block = (threads_per_block + 63) / 64;
    constexpr size_t data_size = no_blocks * (messages_per_wave * waves_per_block * sizeof(dh_comms::wave_header_t) +
                                              messages_per_wave * threads_per_block *
                                                  (sizeof(dh_comms::lane_header_t) + sizeof(float *)));

    float *src, *dst;
    CHK_HIP_ERR(hipMalloc(&src, size * sizeof(float)));
    CHK_HIP_ERR(hipMalloc(&dst, size * sizeof(float)));

    // dh_comms::dh_comms configuration parameters
    constexpr size_t no_sub_buffers = 256;
    constexpr size_t sub_buffer_capacity = 64 * 1024;
    constexpr size_t no_host_threads = 1;
    // memory heatmap configuration parameters
    constexpr size_t page_size = 1024 * 1024;
    constexpr bool verbose = false;
    dh_comms::memory_heatmap_t memory_heatmap(page_size, verbose);

    hipEvent_t start, stop;
    CHK_HIP_ERR(hipEventCreate(&start));
    CHK_HIP_ERR(hipEventCreate(&stop));
    {
        dh_comms::dh_comms dh_comms(no_sub_buffers, sub_buffer_capacity, memory_heatmap, no_host_threads, verbose);
        CHK_HIP_ERR(hipDeviceSynchronize());
        CHK_HIP_ERR(hipEventRecord(start));
        // if dh_comms sub-buffers get full during running of the kernel,
        // device code notifies host code to process the full buffers and
        // clear them
        test<<<no_blocks, threads_per_block>>>(dst, src, 3.14, size, dh_comms.get_dev_rsrc_ptr());

        // dh_comms::dh_comms destructor waits for all kernels
        // to finish, and then processes any remaining data in the
        // sub-buffers
    }
    CHK_HIP_ERR(hipEventRecord(stop));
    CHK_HIP_ERR(hipEventSynchronize(stop));
    float ms;
    CHK_HIP_ERR(hipEventElapsedTime(&ms, start, stop));
    float mbps = (float)data_size / ms / 1000;
    printf("processed %lu bytes in %f ms: %.0f MiB/s\n", data_size, ms, mbps);

    CHK_HIP_ERR(hipFree(src));
    CHK_HIP_ERR(hipFree(dst));

    printf("\n");
    memory_heatmap.show();
}