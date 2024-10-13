#include "message_handlers.h"

namespace dh_comms
{
    message_handler_base::~message_handler_base() {}

    bool message_handlers_t::handle(const message_t &message)
    {
        for (auto &mh : message_handlers)
        {
            if (mh->handle(message))
            {
                return true;
            }
        }
        return false;
    }

    void message_handlers_t::add_handler(std::unique_ptr<message_handler_base> &&message_handler)
    {
        message_handlers.push_back(std::move(message_handler));
    }

} // namespace dh_comms