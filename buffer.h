#pragma once
#include <cstddef>

namespace dh_comms
{
    class buffer
    {
    public:
        buffer(std::size_t packets_per_sub_buffer);
        ~buffer();

    private:
        void *buffer_ = nullptr;
        std::size_t no_sub_buffers_ = 0;
    };

} // namespace dh_comms