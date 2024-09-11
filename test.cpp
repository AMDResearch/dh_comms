#include "buffer.h"
#include "packet.h"
#include "hip_utils.h"
#include <hip/hip_runtime.h>

__global__ void test()
{
    uint16_t cu_id = dh_comms::get_cu_id();
    uint16_t xcc = (cu_id & 0x780) >> 7;
    uint16_t se = (cu_id & 0x70) >> 4;
    uint16_t cu = (cu_id & 0xf);
    uint16_t value = xcc + se + cu + threadIdx.x + blockIdx.x;
    dh_comms::packet p;
    fill_packet(p, true, true, value);
    /*
    printf("[Device]  [%d%d] thread [%d:%d:%d] block [%d:%d:%d] on CU [%02u:%02u:%02u] submitting value %u\n",
           true, true,
           p.thread_x, p.thread_y, p.thread_z,
           p.wg_x, p.wg_y, p.wg_z,
           xcc, se, cu, p.value);
           */
    submit_packet(p);
}

int main()
{
    constexpr size_t packets_per_sub_buffer = 1024;
    constexpr int no_blocks = 1024 * 104 * 16 + 3;
    constexpr size_t data_size = no_blocks * 64 * sizeof(dh_comms::packet);

    hipEvent_t start, stop;
    CHK_HIP_ERR(hipEventCreate(&start));
    CHK_HIP_ERR(hipEventCreate(&stop));
    {
        dh_comms::buffer buffer(packets_per_sub_buffer);
        CHK_HIP_ERR(hipDeviceSynchronize());
        CHK_HIP_ERR(hipEventRecord(start));
        test<<<no_blocks, 64>>>();
    }
    CHK_HIP_ERR(hipEventRecord(stop));
    CHK_HIP_ERR(hipEventSynchronize(stop));
    float ms;
    CHK_HIP_ERR(hipEventElapsedTime(&ms, start, stop));
    float mbps = (float)data_size / ms / 1000;
    printf("processed %lu bytes in %f ms: %.0f MiB/s\n", data_size, ms, mbps);
}