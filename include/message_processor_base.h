#pragma once

namespace dh_comms
{
    class message_processor_base
    {
    public:
        virtual ~message_processor_base() {};
        virtual size_t operator()(char *&message_p, size_t size, size_t sub_buf_no) = 0;
        virtual bool is_thread_safe() const { return false; }
    };
} // namespace dh_comms