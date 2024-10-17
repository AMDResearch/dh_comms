#pragma once

#include <memory>
#include <vector>
#include "message.h"

namespace dh_comms
{
    //! \brief base class for message handlers on the host.
    //!
    //! dh_comms maintains one or more chains of message handlers, one chain per
    //! host processing thread. The message handlers in a chain get to look at the
    //! message, and determine whether they can handle it by inspecting the wave header,
    //! and in most cases, the user_type field of the wave header in particular. If a
    //! message handler cannot handle the message, its handle() function returns false,
    //! which will result in the next handler in the chain to get a chance at handling
    //! the message. Otherwise, if a handler can handle a message, it does so, and its
    //! handle() function returns true, which stops the message from being considered
    //! by further message handlers down the chain.
    class message_handler_base
    {
    public:
        message_handler_base(){};
        message_handler_base(const message_handler_base&) = default;
        virtual ~message_handler_base() = 0;
        //! A derived class implementing handle() must handle a message if it can and return
        //! true, or, if it cannot handle the message, return false.
        virtual bool handle(const message_t& message) = 0;
        //! Since there can be multiple message handler chains, the handlers of a particular
        //! particular message type potentially store aggregate data in their data structures.
        //! For instance, consider two handler chains, each with a memory map handler that stores
        //! page counts in a std::map, where each of the two handlers processes roughly half of
        //! the messages containing addresses. Once processing is done, these two std::maps need
        //! to be merged into a single map. In the example, this can be done by inserting all
        //! the items in the std::map of "other" into the std::map of "this", and then clearing the
        //! std::map of other.
        //! Not all message handlers may have state that needs to be merged. In such cases, overriding
        //! merge_state in the derived class isn't necessary, and the derived class just inherits the
        //! base class implementation, which does nothing.
        virtual void merge_state(message_handler_base&){};
        //! Message handlers that aggregate data during message processing may want to report
        //! the data when done. They may do so by overriding this function. Not all message handlers
        //! may need to report data in the end, e.g., message handlers that just save messages to disk
        //! on the fly as they are processed. These handlers do not need override this function, but just
        //! rely on the implementation of this function by the base class, which does nothing.
        virtual void report(){}
        //! Stateful message handlers must implement the clear function by clearing their state,
        //! so that they can be reused for a new data processing run. Stateless message handlers
        //! don't need to override this function, but can inherit the base class implementation.
        virtual void clear(){};
        //! Message handlers are implemented as classes derived from the abstract base class
        //! message_handler_base. The classes that manage the message handlers only use (smart) pointers
        //! to message_handler_base, but they need to be able to copy message handlers. We use the
        //! standard solution of requiring the derived classes to implement a clone function,
        //! clone_impl(). For classes with a copy constructor, the implementation of clone_impl
        //! can be a one-liner; see for example memory_heatmap_t::clone_impl() in memory_heatmap.h.
        auto clone() const { return std::unique_ptr<message_handler_base>(clone_impl()); }
    protected:
        //! Objects of derived classes are required to return a pointer to a heap-based copy of themselves,
        //! e.g., by "return new Derived(*this);"
        virtual message_handler_base* clone_impl() const = 0;
    };

    class message_handler_chain_t {
    public:
        message_handler_chain_t();
        ~message_handler_chain_t() = default;
        message_handler_chain_t(const message_handler_chain_t&) = delete;
        message_handler_chain_t& operator=(const message_handler_chain_t&) = delete;

        size_t size() const;
        bool handle(const message_t& message);
        size_t bytes_processed() const;
        void add_handler(std::unique_ptr<message_handler_base>&& message_handler);
        void report();
        void clear_handler_states();
        void merge_handler_states(message_handler_chain_t& other);

    private:
        std::vector<std::unique_ptr<message_handler_base>> message_handlers_;
        size_t bytes_processed_;
    };
} // namespace dh_comms