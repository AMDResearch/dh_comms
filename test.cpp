#include "buffer.h"
#include "data_headers.h"
#include "hip_utils.h"
#include <hip/hip_runtime.h>

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
size_t process_sub_buffer(char *&message_p, size_t size, size_t sub_buf_no)
{
    using dh_comms::lane_header_t;
    using dh_comms::wave_header_t;

    printf("[Host] %zu bytes of data remaining in sub-buffer %zu\n", size, sub_buf_no);
    wave_header_t *wave_header_p = (wave_header_t *)message_p;
    size_t data_size = wave_header_p->data_size;
    printf("wave_header:\n");
    printf("\texec = 0x%016lx\n", wave_header_p->exec);
    uint32_t active_lane_count = wave_header_p->active_lane_count;
    printf("\tactive_lane_count = %u\n", active_lane_count);
    printf("\t[block]:wave = [%u,%u,%u]:%u\n", wave_header_p->block_idx_x, wave_header_p->block_idx_y,
           wave_header_p->block_idx_z, wave_header_p->wave_id);
    printf("\txcc:se:cu = %02u:%02u:%02u\n", wave_header_p->xcc_id, wave_header_p->se_id,
           wave_header_p->cu_id);
    message_p += sizeof(wave_header_t);
    lane_header_t *lane_header_p = (lane_header_t *)message_p;
    for (uint32_t lane = 0; lane != active_lane_count; ++lane)
    {
        printf("\t[thread] = [%u,%u,%u]\n", lane_header_p->thread_idx_x,
               lane_header_p->thread_idx_y, lane_header_p->thread_idx_z);
        ++lane_header_p;
    }
    message_p += data_size;
    size -= (sizeof(wave_header_t) + data_size);
    printf("\n");

    return size;
}

int main()
{
    // dh_comms::buffer configuration parameters
    constexpr size_t no_sub_buffers = 256;
    constexpr size_t sub_buffer_capacity = 64 * 1024;
    constexpr size_t no_host_threads = 1;

    // kernel launch parameters
    constexpr int no_blocks = 3; // 1024 * 128 * 16 + 3;
    constexpr int threads_per_block = 128;
    // constexpr size_t data_size = no_blocks * 64 * sizeof(dh_comms::packet);

    hipEvent_t start, stop;
    CHK_HIP_ERR(hipEventCreate(&start));
    CHK_HIP_ERR(hipEventCreate(&stop));
    {
        dh_comms::buffer buffer(no_sub_buffers, sub_buffer_capacity, process_sub_buffer, no_host_threads);
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
    // float mbps = (float)data_size / ms / 1000;
    // printf("processed %lu bytes in %f ms: %.0f MiB/s\n", data_size, ms, mbps);
}