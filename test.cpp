#include <hip/hip_runtime.h>
#include "buffer.h"
#include "hip_utils.h"
#include "memory_heatmap.h"


__global__ void test()
{
    // we can submit any type of message; here, we're using an uint64_t
    // (could be a memory address in real code)
    uint64_t message = 42 + threadIdx.x;
    // since our buffers can have a mix of multiple types of messages,
    // we need to tag messages with a message type, so that
    // host processing code knows what to do with the message
    uint32_t user_defined_message_type = 0;

    // submit from four active lanes in the first wave and one in the
    // second wave to test whether we properly handle active/inactive lanes
    if ((threadIdx.x > 3 and threadIdx.x < 8) or threadIdx.x == 120)
    {
        dh_comms::submit_message(message, user_defined_message_type);
    }
}


int main()
{
    // dh_comms::buffer configuration parameters
    constexpr size_t no_sub_buffers = 256;
    constexpr size_t sub_buffer_capacity = 64 * 1024;
    constexpr size_t no_host_threads = 1;
    // memory heatmap configuration parameters
    constexpr size_t page_size = 5;
    constexpr bool verbose = false;
    dh_comms::memory_heatmap_t memory_heatmap(page_size, verbose);

    // kernel launch parameters
    constexpr int no_blocks = 1024 * no_sub_buffers;
    constexpr int threads_per_block = 128;
    constexpr size_t data_size = no_blocks * (2 * sizeof(dh_comms::wave_header_t) + 5 * sizeof(dh_comms::lane_header_t) + 5 * sizeof(uint64_t));

    hipEvent_t start, stop;
    CHK_HIP_ERR(hipEventCreate(&start));
    CHK_HIP_ERR(hipEventCreate(&stop));
    {
        dh_comms::buffer buffer(no_sub_buffers, sub_buffer_capacity, memory_heatmap, no_host_threads);
        CHK_HIP_ERR(hipDeviceSynchronize());
        CHK_HIP_ERR(hipEventRecord(start));
        // if dh_comms sub-buffers get full during running of the kernel,
        // device code notifies host code to process the full buffers and
        // clear them
        test<<<no_blocks, threads_per_block>>>();

        // dh_comms::buffer destructor waits for all kernels
        // to finish, and then processes any remaining data in the
        // sub-buffers
    }
    CHK_HIP_ERR(hipEventRecord(stop));
    CHK_HIP_ERR(hipEventSynchronize(stop));
    float ms;
    CHK_HIP_ERR(hipEventElapsedTime(&ms, start, stop));
    float mbps = (float)data_size / ms / 1000;
    printf("processed %lu bytes in %f ms: %.0f MiB/s\n", data_size, ms, mbps);

    printf("\n");
    memory_heatmap.show();
}