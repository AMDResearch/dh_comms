#pragma once
#include <cstddef>
#include <vector>
#include <thread>
#include <functional>

#include "data_headers.h"
#include "message_processor_base.h"

namespace dh_comms
{
    //! \brief Keeps track of resources used by device and host code for exchanging data.
    //!
    //! The main data buffer is partitioned into a number of equal-sized sub-buffers to
    //! allow for concurrent access. Multiple waves can write to different sub-buffers
    //! simultaneously. Waves that want to write to the same sub-buffer are serialized
    //! using atomics.
    struct dh_comms_resources
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

        //! Constructor; allocates memory resources based on its arguments.
        dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity);
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
                 message_processor_base &message_processor, //!<  Pointer to derived class of message_processor_base, responsible
                                                            //!< for processing messages by host code. Since dh_comms users can submit any type of
                                                            //!< message from device code to the host, interpretation of the data on the host
                                                            //!< side is the responsibility of user code too.
                 std::size_t no_host_threads = 1,           //!< Controls how many threads host code uses to process messages in the
                                                            //!< sub-buffers.
                 bool verbose = false                       //!< Controls how chatty the code is.
                 );
        ~dh_comms();
        dh_comms(const dh_comms &) = delete;
        dh_comms &operator=(const dh_comms &) = delete;
        dh_comms_resources *get_dev_rsrc_ptr();             //!< Returns a pointer to a dh_comms_resources struct in device memory.

    private:
        void process_sub_buffers(std::size_t first, std::size_t last);
        std::vector<std::thread> init_host_threads(std::size_t no_host_threads,
                                                   bool message_processor_is_thread_safe);

    private:
        dh_comms_resources rsrc_;
        dh_comms_resources *dev_rsrc_p_;
        std::vector<std::thread> sub_buffer_processors_;
        message_processor_base &message_processor_;
        volatile bool teardown_;
        bool verbose_;
    };

} // namespace dh_comms