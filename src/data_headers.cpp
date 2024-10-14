#include "data_headers.h"
#include <cstring>

namespace dh_comms
{
    wave_header_t::wave_header_t(const char *wave_header_p)
    {
        memcpy((char *)this, wave_header_p, sizeof(wave_header_t));
        /*
        printf("wave_header:\n");
        printf("\tcopied from address %p\n", wave_header_p);
        printf("\texec = 0x%016lx\n", exec);
        printf("\tdata size = %lu\n", data_size);
        printf("\t%s message %s lane headers\n", is_vector_message ? "vector" : "scalar",
               has_lane_headers ? "with" : "without");
        printf("\ttimestamp = %lu\n", timestamp);
        printf("\tsrc_loc_idx = 0x%x\n", src_loc_idx);
        printf("\tuser_type = 0x%x\n", user_type);
        printf("\tactive_lane_count = %u\n", active_lane_count);
        printf("\t[block]:wave = [%u,%u,%u]:%u\n", block_idx_x, block_idx_y,
               block_idx_z, wave_num);
        printf("\txcc:se:cu = %02u:%02u:%02u\n", xcc_id, se_id,
               cu_id);
        */
    }

    lane_header_t::lane_header_t(const char *lane_header_p)
    {
        memcpy((char *)this, lane_header_p, sizeof(lane_header_t));
    }
}