#pragma once

#include "message.h"

#include <memory>
#include <vector>

namespace dh_comms {
//! \brief base class for message handlers on the host.
//!
//! dh_comms maintains a chain of message handlers. These message handlers get to look at
//! the message, and determine whether they can handle it by inspecting the wave header,
//! and in most cases, the user_type field of the wave header in particular. If a
//! message handler cannot handle the message, its handle() function returns false,
//! which will result in the next handler in the chain to get a chance at handling
//! the message. Otherwise, if a handler can handle a message, it does so, and its
//! handle() function returns true, which stops the message from being considered
//! by further message handlers down the chain.
class message_handler_base {
public:
  message_handler_base() {};
  message_handler_base(const message_handler_base &) = default;
  virtual ~message_handler_base() = 0;
  //! A derived class implementing handle() must handle a message if it can and return
  //! true, or, if it cannot handle the message, return false.
  virtual bool handle(const message_t &message) = 0;
  //! Stateful message handlers that aggregate data during message processing may want to report
  //! the data when done. They may do so by overriding this function. Not all message handlers
  //! may need to report data in the end, e.g., stateless message handlers that just save messages to disk
  //! on the fly as they are processed. These handlers do not need override this function, but just
  //! rely on the implementation of this function by the base class, which does nothing.
  virtual void report() {}
  //! Stateful message handlers must implement the clear function by clearing their state,
  //! so that they can be reused for a new data processing run. Stateless message handlers
  //! don't need to override this function, but can inherit the base class implementation.
  virtual void clear() {};
};

class message_handler_chain_t {
public:
  message_handler_chain_t(bool pass_through = false);
  ~message_handler_chain_t() = default;
  message_handler_chain_t(const message_handler_chain_t &) = delete;
  message_handler_chain_t(message_handler_chain_t &&) = default;
  message_handler_chain_t &operator=(const message_handler_chain_t &) = delete;

  size_t size() const;
  bool handle(const message_t &message);
  void add_handler(std::unique_ptr<message_handler_base> &&message_handler);
  void report();
  void clear_handler_states(); //!< Keeps the handlers, but clears their states
  void clear();                //!< Removes all handlers from the chain

private:
  std::vector<std::unique_ptr<message_handler_base>> message_handlers_;
  bool pass_through_;
};
} // namespace dh_comms