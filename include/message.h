#pragma once

#include <vector>
#include "data_headers.h"

namespace dh_comms
{
    enum class e_message: uint32_t
    {
        address = 0,
        undefined = 0xffffffff
    };

    struct message_t
    {
        wave_header_t wave_header;
        std::vector<lane_header_t> lane_headers;
        std::vector <std::vector<char>> data;

        message_t(const char *message_p);
        size_t size() const;
    };
}