#pragma once
#include <cstddef>
#include <vector>

namespace dh_comms
{
    class buffer
    {
    public:
        buffer(std::size_t packets_per_sub_buffer);
        ~buffer();
        buffer(const buffer&) = delete;
        buffer& operator=(const buffer&) = delete;

        void print_cu_to_index_map() const;

    private:
        std::size_t no_sub_buffers_ = 0;        // Current implementation: 1 sub-buffer per CU
        std::vector<uint16_t> index_to_cu_map_; // maps 11 bits CU ids (XCC|SE|CU) to 0..#CUs - 1
        std::vector<uint16_t> cu_to_index_map_; // inverse of the above map
        void *buffer_ = nullptr;
    };

} // namespace dh_comms