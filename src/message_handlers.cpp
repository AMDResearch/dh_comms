#include "message_handlers.h"

#include <cassert>

namespace dh_comms {
message_handler_base::~message_handler_base() {}

message_handler_chain_t::message_handler_chain_t(bool pass_through)
    : pass_through_(pass_through) {}

size_t message_handler_chain_t::size() const { return message_handlers_.size(); }

bool message_handler_chain_t::handle(const message_t &message) {
  for (auto &mh : message_handlers_) {
    if (mh->handle(message) and not pass_through_) {
      return true;
    }
  }
  return false;
}

void message_handler_chain_t::add_handler(std::unique_ptr<message_handler_base> &&message_handler) {
  message_handlers_.push_back(std::move(message_handler));
}

void message_handler_chain_t::report() {
  for (auto &mh : message_handlers_) {
    mh->report();
  }
}

void message_handler_chain_t::clear_handler_states() {
  for (auto &mh : message_handlers_) {
    mh->clear();
  }
}

void message_handler_chain_t::clear() { message_handlers_.clear(); }

} // namespace dh_comms