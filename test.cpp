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
    printf("[Device]  [%d%d] thread [%d:%d:%d] block [%d:%d:%d] on CU [%02u:%02u:%02u] submitting value %u\n",
           true, true,
           p.thread_x, p.thread_y, p.thread_z,
           p.wg_x, p.wg_y, p.wg_z,
           xcc, se, cu, p.value);
    submit_packet(p);
}

int main()
{
    constexpr size_t packets_per_sub_buffer = 1024;
    dh_comms::buffer buffer(packets_per_sub_buffer);

    // buffer.print_cu_to_index_map();
    test<<<3, 2>>>();
    CHK_HIP_ERR(hipDeviceSynchronize());
    buffer.show_queues();
}