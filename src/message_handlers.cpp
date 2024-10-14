#include "message_handlers.h"
#include <cassert>

namespace dh_comms
{
    message_handler_base::~message_handler_base() {}

    bool message_handlers_t::handle(const message_t &message)
    {
        bytes_processed_ += message.size();
        for (auto &mh : message_handlers_)
        {
            if (mh->handle(message))
            {
                return true;
            }
        }
        return false;
    }

    message_handlers_t::message_handlers_t()
    : bytes_processed_(0)
    {}

    size_t message_handlers_t::bytes_processed() const
    {
        return bytes_processed_;
    }

    void message_handlers_t::add_handler(std::unique_ptr<message_handler_base> &&message_handler)
    {
        message_handlers_.push_back(std::move(message_handler));
    }

    void message_handlers_t::merge_handler_states(message_handlers_t &other)
    {
        assert(message_handlers_.size() == other.message_handlers_.size());
        for (size_t i = 0; i < message_handlers_.size(); ++i)
        {
            message_handlers_[i]->merge_state(*other.message_handlers_[i]);
        }
    }

} // namespace dh_comms