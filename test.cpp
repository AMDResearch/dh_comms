#include "buffer.h"
#include <hip/hip_runtime.h>

__global__ void test_constants()
{
    if(blockIdx.x == 0)
    {
        dh_comms::test_constants_f();
    }
    if(threadIdx.x == 0)
    {
        for(uint16_t i=0; i != 64; ++i)
        {
            printf("%u -> %lu\n", i, dh_comms::cu_to_index_map_f(i));
        }
    }
}



int main(){
    constexpr size_t packets_per_sub_buffer = 1024;
    dh_comms::buffer buffer(packets_per_sub_buffer);

    buffer.print_cu_to_index_map();
    test_constants<<<1,1>>>();
}