#pragma once

#include <vector>
#include "data_headers.h"

namespace dh_comms
{
    enum class message_type : uint32_t
    {
        address = 0,
        undefined = 0xffffffff
    };

    class message_t
    {
    public:
        message_t(const char *message_p);
        size_t size() const;

        const wave_header_t &wave_header() const { return wave_header_; }
        size_t no_lane_headers() const { return lane_headers_.size(); }
        size_t no_data_items() const { return data_.size(); }
        size_t data_item_size() const { return data_.size() == 0 ? 0 : data_[0].size(); }
        const lane_header_t &lane_header(size_t i) const { return lane_headers_[i]; }
        const void *data_item(size_t i) const { return data_[i].data(); }

    private:
        wave_header_t wave_header_;
        std::vector<lane_header_t> lane_headers_;
        std::vector<std::vector<char>> data_;
    };
}