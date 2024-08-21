#include "packet.h"
#include <hip/hip_runtime.h>

namespace dh_comms {

    __device__ void fill_packet(packet& p, bool is_first, bool is_last, uint16_t value){
        p.is_first = is_first;
        p.is_last = is_last;
        p.thread_x = threadIdx.x;
        p.thread_y = threadIdx.y;
        p.thread_z = threadIdx.z;
        p.wg_x = blockIdx.x;
        p.wg_y = blockIdx.y;
        p.wg_z = blockIdx.z;
        p.value = value;
    }

    std::string packet_str(const packet& p){
        std::ostringstream oss;
        oss << "[" << p.is_first << p.is_last << "] thread ["
        << p.thread_x << ":" << p.thread_y << ":" << p.thread_z << "] block ["
        << p.wg_x << ":" << p.wg_y << ":" << p.wg_z << "] value " << p.value;
        return oss.str();
    }

} // namespace dh_comms