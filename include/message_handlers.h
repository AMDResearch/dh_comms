#pragma once

#include <memory>
#include <vector>
#include "message.h"

namespace dh_comms
{
    class message_handler_base
    {
    public:
        message_handler_base(){};
        virtual ~message_handler_base() = 0;
        virtual bool handle(const message_t& message) = 0;
        virtual void merge_state(message_handler_base&){};
    };

    class message_handlers_t {
    public:
        message_handlers_t();
        ~message_handlers_t() = default;
        message_handlers_t(const message_handlers_t&) = delete;
        message_handlers_t& operator=(const message_handlers_t&) = delete;

        bool handle(const message_t& message);
        size_t bytes_processed() const;
        void add_handler(std::unique_ptr<message_handler_base>&& message_handler);
        void merge_handler_states(message_handlers_t& other);

    private:
        std::vector<std::unique_ptr<message_handler_base>> message_handlers_;
        size_t bytes_processed_;
    };
} // namespace dh_comms