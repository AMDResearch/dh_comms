#include "buffer.h"

int main(){
    constexpr size_t packets_per_sub_buffer = 1024;
    dh_comms::buffer buffer(packets_per_sub_buffer);
    buffer.print_cu_to_index_map();
}