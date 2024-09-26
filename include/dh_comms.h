#pragma once
#include <cstddef>
#include <vector>
#include <thread>
#include <functional>

#include "data_headers.h"
#include "hip_utils.h"
#include "message_processor_base.h"

namespace dh_comms
{
    struct dh_comms_resources
    {
        std::size_t no_sub_buffers_;
        std::size_t sub_buffer_capacity_;
        char *buffer_;
        size_t *sub_buffer_sizes_;
        uint32_t* error_bits_;
        uint8_t *atomic_flags_;

        dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity);
        dh_comms_resources(const dh_comms_resources &) = delete;
        dh_comms_resources &operator=(const dh_comms_resources &) = delete;
        ~dh_comms_resources();
    };

    class dh_comms
    {
    public:
        dh_comms(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity,
                 message_processor_base &message_processor,
                 std::size_t no_host_threads = 1);
        ~dh_comms();
        dh_comms(const dh_comms &) = delete;
        dh_comms &operator=(const dh_comms &) = delete;

        dh_comms_resources *get_dev_rsrc_ptr();

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
    };

} // namespace dh_comms