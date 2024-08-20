#include "buffer.h"

int main(){
    dh_comms::buffer buffer(1024);
    buffer.print_cu_to_index_map();
}