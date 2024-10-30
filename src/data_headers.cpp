#include "data_headers.h"
#include <cstring>

namespace dh_comms
{
    wave_header_t::wave_header_t(const char *wave_header_p)
    {
        memcpy((char *)this, wave_header_p, sizeof(wave_header_t));
    }

    lane_header_t::lane_header_t(const char *lane_header_p)
    {
        memcpy((char *)this, lane_header_p, sizeof(lane_header_t));
    }
}