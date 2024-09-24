#pragma once

namespace dh_comms
{
    class message_processor_base
    {
    public:
        virtual ~message_processor_base() {};
        virtual size_t operator()(char *&message_p, size_t size, size_t sub_buf_no) = 0;
    };
} // namespace dh_comms