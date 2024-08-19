#include <string>
#include <cstdio>

#include <hip/hip_runtime.h>
#include "hip_utils/hip_utils.h"

#include "buffer.h"
#include "packet.h"

namespace {
    size_t get_cu_count(){
        hipDeviceProp_t props;
        CHK_HIP_ERR(hipGetDeviceProperties(&props, 0)); // TODO: handle multiple devices
        return props.multiProcessorCount;
    }
} // unnamed namespace

namespace dh_comms {

buffer::buffer(std::size_t packets_per_sub_buffer){
    std::size_t cu_count = get_cu_count();
    std::size_t size = cu_count * packets_per_sub_buffer * bytes_per_packet;
    printf("Allocating %lu bytes for %lu CUs\n", size, cu_count);
    CHK_HIP_ERR(hipMalloc(&buffer_, size));
}

buffer::~buffer(){
    CHK_HIP_ERR(hipFree(buffer_));
}

} // namespace dh_comms