#pragma once
#include <cstddef>
#include <vector>
#include <thread>
#include <functional>
#include <chrono>

#include "data_headers.h"
#include "message_handlers.h"

namespace dh_comms
{
    class dh_comms_mem_mgr {
    public:
        dh_comms_mem_mgr();
        virtual ~dh_comms_mem_mgr();
        virtual void * alloc(std::size_t size);
        virtual void * alloc_device_memory(std::size_t size);
        virtual void free(void *);
        virtual void free_device_memory(void *);
        virtual void * copy(void *dst, void *src, std::size_t size);
        virtual void * copy_to_device(void *dst, const void *src, std::size_t size);
        virtual std::size_t zero(void *buffer, std::size_t size);
    };

    struct dh_comms_descriptor
    {
        std::size_t no_sub_buffers_;        //!< Number of sub-buffers into which the main data buffer is partitioned.
        std::size_t sub_buffer_capacity_;   //!< The maximum number of bytes each of the sub-buffers can hold.
        char *buffer_;                      //!< Pointer to the main data buffer.
        size_t *sub_buffer_sizes_;          //!< Pointer to an array of no_sub_buffers_ entries. Each entry holds the number
                                            //!< of bytes (0 <= size <= sub_buffer_capacity_) currently in the corresponding
                                            //!< sub-buffer.
        uint32_t* error_bits_;              //!< Pointer to an array of no_sub_buffers_ entries, one for each sub-buffer,
                                            //!< used to track error conditions for the sub-buffer, such as a wave attempting
                                            //!< to write more than sub_buffer_capacity_ bytes to a sub-buffer.
        uint8_t *atomic_flags_;
    };
    //! \brief Keeps track of resources used by device and host code for exchanging data.
    //!
    //! The main data buffer is partitioned into a number of equal-sized sub-buffers to
    //! allow for concurrent access. Multiple waves can write to different sub-buffers
    //! simultaneously. Waves that want to write to the same sub-buffer are serialized
    //! using atomics.
    struct dh_comms_resources
    {
        dh_comms_descriptor desc_;
        dh_comms_mem_mgr& mgr_;

        //! Constructor; allocates memory resources based on its arguments.
        dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity, dh_comms_mem_mgr& mgr);
        dh_comms_resources(const dh_comms_resources &) = delete;
        dh_comms_resources &operator=(const dh_comms_resources &) = delete;
        //! Destructor; releases allocated memory.
        ~dh_comms_resources();
    };


    //! \brief Orchestrates allocation of resources for message passing from device to host code
    //! and processing of messages on the host
    class dh_comms
    {
    public:
        dh_comms(std::size_t no_sub_buffers,                //!< Number of sub-buffers into which the main data buffer is partitioned.
                 std::size_t sub_buffer_capacity,           //!< The maximum number of bytes each of the sub-buffers can hold.
                 // message_processor_base &message_processor, //!<  Pointer to derived class of message_processor_base, responsible
                                                            //!< for processing messages by host code. Since dh_comms users can submit any type of
                                                            //!< message from device code to the host, interpretation of the data on the host
                                                            //!< side is the responsibility of user code too.
                 std::size_t no_host_threads = 1,           //!< Controls how many threads host code uses to process messages in the
                                                            //!< sub-buffers.
                 bool verbose = false,                       //!< Controls how chatty the code is.
                 dh_comms_mem_mgr *mgr = NULL
                 );
        ~dh_comms();
        dh_comms(const dh_comms &) = delete;
        dh_comms &operator=(const dh_comms &) = delete;
        dh_comms_descriptor *get_dev_rsrc_ptr();             //!< Returns a pointer to a dh_comms_resources struct in device memory.

        void start();                                        //! Start the message processing threads on the host.
        void stop();                                         //! Stop message processing on the host. It is the responsibility
                                                             //! of calling code to make sure kernels have finished by e.g. issuing
                                                             //! a hipDeviceSyncronize() or other synchronization call.
        void clear_handler_states();
        void merge_handler_states();
        void report(bool auto_merge = true,
                    bool auto_clear = true);

    private:
        void process_sub_buffers(std::size_t thread_no, std::size_t first, std::size_t last);
        std::vector<message_handler_chain_t> init_message_handler_chains(std::size_t no_host_threads);

    private:
        dh_comms_mem_mgr default_mgr_;
        dh_comms_mem_mgr * mgr_;
        dh_comms_resources rsrc_;
        dh_comms_descriptor *dev_rsrc_p_;
        std::size_t no_host_threads_;
        std::vector<message_handler_chain_t> message_handler_chains_;
        std::vector<std::thread> sub_buffer_processors_;
        // message_processor_base &message_processor_;
        volatile bool running_;
        const bool verbose_;
        std::chrono::time_point<std::chrono::steady_clock> start_time_;
        std::chrono::time_point<std::chrono::steady_clock> stop_time_;
    };

} // namespace dh_comms
