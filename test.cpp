#include "buffer.h"
#include "data_headers.h"
#include "hip_utils.h"
#include <hip/hip_runtime.h>

__global__ void test()
{
    size_t message = 42 + threadIdx.x;
    uint32_t user_defined_message_type = 0;
    if((threadIdx.x > 3 and threadIdx.x < 8) or threadIdx.x == 120)
    {
        dh_comms::submit_message(message, user_defined_message_type);
    }
}

int main()
{
    constexpr size_t no_sub_buffers = 256;
    constexpr size_t sub_buffer_capacity = 64 * 1024;
    constexpr size_t no_host_threads = 1;
    constexpr int no_blocks = 3; // 1024 * 128 * 16 + 3;
    constexpr int threads_per_block = 128;
    // constexpr size_t data_size = no_blocks * 64 * sizeof(dh_comms::packet);

    hipEvent_t start, stop;
    CHK_HIP_ERR(hipEventCreate(&start));
    CHK_HIP_ERR(hipEventCreate(&stop));
    {
        dh_comms::buffer buffer(no_sub_buffers, sub_buffer_capacity, no_host_threads);
        CHK_HIP_ERR(hipDeviceSynchronize());
        CHK_HIP_ERR(hipEventRecord(start));
        test<<<no_blocks, threads_per_block>>>();
    }
    CHK_HIP_ERR(hipEventRecord(stop));
    CHK_HIP_ERR(hipEventSynchronize(stop));
    float ms;
    CHK_HIP_ERR(hipEventElapsedTime(&ms, start, stop));
    // float mbps = (float)data_size / ms / 1000;
    // printf("processed %lu bytes in %f ms: %.0f MiB/s\n", data_size, ms, mbps);
}