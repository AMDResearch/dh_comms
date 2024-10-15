#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

#include <hip/hip_runtime.h>
#include "hip_utils.h"

#include "dh_comms.h"
#include "data_headers.h"
#include "message.h"
#include "memory_heatmap.h"

namespace dh_comms
{

    dh_comms_mem_mgr::dh_comms_mem_mgr()
    {
        return;
    }

    dh_comms_mem_mgr::~dh_comms_mem_mgr()
    {
    }

    void *dh_comms_mem_mgr::alloc(std::size_t size)
    {
        void *buffer;
        CHK_HIP_ERR(hipHostMalloc(&buffer, size, hipHostMallocCoherent));
        zero((char *)buffer, size);
        return buffer;
    }

    void *dh_comms_mem_mgr::alloc_device_memory(std::size_t size)
    {
        void *result = NULL;
        CHK_HIP_ERR(hipMalloc(&result, size));
        return result;
    }

    void *dh_comms_mem_mgr::copy_to_device(void *dst, const void *src, std::size_t size)
    {
        CHK_HIP_ERR(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
        return dst;
    }

    void dh_comms_mem_mgr::free(void *ptr)
    {
        CHK_HIP_ERR(hipFree(ptr));
        return;
    }

    void dh_comms_mem_mgr::free_device_memory(void *ptr)
    {
        this->free(ptr);
    }

    void *dh_comms_mem_mgr::copy(void *dst, void *src, std::size_t size)
    {
        memcpy(dst, src, size);
        return dst;
    }

    std::size_t dh_comms_mem_mgr::zero(void *buffer, std::size_t size)
    {
        std::vector<char> zeros(size);
        std::copy(zeros.cbegin(), zeros.cend(), (char *)buffer);
        return size;
    }
}

namespace
{
    constexpr bool shared_buffers_are_host_pinned = true;

    /*void *allocate_shared_buffer(size_t size)
    {
        void *buffer;
        std::vector<char> zeros(size);
        if constexpr (shared_buffers_are_host_pinned)
        {
            CHK_HIP_ERR(hipHostMalloc(&buffer, size, hipHostMallocCoherent));
            std::copy(zeros.cbegin(), zeros.cend(), (char *)buffer);
        }
        else
        {
            CHK_HIP_ERR(hipExtMallocWithFlags(&buffer, size, hipDeviceMallocFinegrained));
            CHK_HIP_ERR(hipMemcpy(buffer, zeros.data(), size, hipMemcpyHostToDevice));
        }
        return buffer;
    }*/

    template <typename T>
    T *clone_to_device(const T &host_data, dh_comms::dh_comms_mem_mgr &mgr)
    {
        T *device_data;
        device_data = reinterpret_cast<T *>(mgr.alloc_device_memory(sizeof(T)));
        mgr.copy_to_device(device_data, &host_data, sizeof(T));
        return device_data;
    }

} // unnamed namespace

namespace dh_comms
{
    dh_comms_resources::dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity, dh_comms_mem_mgr &mgr)
        : desc_({no_sub_buffers, sub_buffer_capacity,
                 (decltype(desc_.buffer_))mgr.alloc(no_sub_buffers * sub_buffer_capacity),
                 (decltype(desc_.sub_buffer_sizes_))mgr.alloc(no_sub_buffers * sizeof(decltype(*desc_.sub_buffer_sizes_))),
                 (decltype(desc_.error_bits_))mgr.alloc(sizeof(decltype(*desc_.error_bits_))),
                 (decltype(desc_.atomic_flags_))mgr.alloc(no_sub_buffers * sizeof(decltype(*desc_.atomic_flags_)))}),
          mgr_(mgr)
    {
    }

    dh_comms_resources::~dh_comms_resources()
    {
        mgr_.free(desc_.atomic_flags_);
        mgr_.free(desc_.error_bits_);
        mgr_.free(desc_.sub_buffer_sizes_);
        mgr_.free(desc_.buffer_);
    }

    dh_comms::dh_comms(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity,
                       /* message_processor_base &message_processor, */
                       std::size_t no_host_threads, bool verbose, dh_comms_mem_mgr *mgr)
        : mgr_(mgr ? mgr : &default_mgr_),
          rsrc_(no_sub_buffers, sub_buffer_capacity, *mgr_),
          dev_rsrc_p_(clone_to_device(rsrc_.desc_, *mgr_)),
          message_handlers_(init_message_handlers(no_host_threads)),
          sub_buffer_processors_(init_host_threads(no_host_threads)),
          // message_processor_(message_processor),
          teardown_(false),
          verbose_(verbose),
          start_time_(std::chrono::steady_clock::now())
    {
        if (verbose_)
        {
            if constexpr (shared_buffers_are_host_pinned)
            {
                printf("%s:%d:\n\t Buffers accessed from both host and device are allocated in pinned host memory\n",
                       __FILE__, __LINE__);
            }
            else
            {
                printf("%s:%d:\n\t Buffers accessed from both host and device are allocated in device memory\n",
                       __FILE__, __LINE__);
            }
            printf("using %zu message handler sets\n", message_handlers_.size());
        }
    }

    dh_comms::~dh_comms()
    {
        CHK_HIP_ERR(hipDeviceSynchronize());
        teardown_ = true;
        for (auto &sbp : sub_buffer_processors_)
        {
            sbp.join();
        }
        const auto stop_time = std::chrono::steady_clock::now();
        if (*rsrc_.desc_.error_bits_ & 1)
        {
            printf("Error detected: data from device dropped because message size was larger than sub-buffer size\n");
        }
        mgr_->free_device_memory(dev_rsrc_p_);
        size_t bytes_processed = 0;
        for (const auto &mh : message_handlers_)
        {
            bytes_processed += mh.bytes_processed();
        }
        const std::chrono::duration<double> processing_time = stop_time - start_time_;
        double MiBps = bytes_processed / processing_time.count() / 1.0e6;
        printf("%zu bytes processed in %lf seconds (%.1lf MiB/s)\n", bytes_processed, processing_time.count(), MiBps);
    }

    dh_comms_descriptor *dh_comms::get_dev_rsrc_ptr()
    {
        return dev_rsrc_p_;
    }

    std::vector<message_handlers_t> dh_comms::init_message_handlers(std::size_t no_host_threads)
    {
        assert(no_host_threads != 0);
        std::vector<message_handlers_t> message_handlers(no_host_threads);
        for (auto &mh : message_handlers)
        {
            mh.add_handler(std::make_unique<memory_heatmap_t>());
        }
        return message_handlers;
    }

    std::vector<std::thread> dh_comms::init_host_threads(std::size_t no_host_threads)
    {
        assert(no_host_threads != 0);
        std::size_t no_sub_buffers_per_thread = rsrc_.desc_.no_sub_buffers_ / no_host_threads;
        std::size_t remainder = rsrc_.desc_.no_sub_buffers_ % no_host_threads;

        std::vector<std::thread> sub_buffer_processors;
        std::size_t first = 0;
        std::size_t last;
        for (std::size_t i = 0; i != no_host_threads; ++i)
        {
            last = first + no_sub_buffers_per_thread;
            if (i < remainder)
            {
                ++last;
            }
            sub_buffer_processors.emplace_back(std::thread(&dh_comms::process_sub_buffers, this, i, first, last));
            first = last;
        }
        assert(last == rsrc_.desc_.no_sub_buffers_);

        return sub_buffer_processors;
    }

    void dh_comms::process_sub_buffers(std::size_t thread_no, std::size_t first, std::size_t last)
    {
        while (__atomic_load_n(&teardown_, __ATOMIC_ACQUIRE) == false)
        {
            for (size_t i = first; i != last; ++i)
            {
                // when the sub-buffer for a wave on the device is full, it will
                // set the flag to either 3 (if it wants control back after the host
                // is done processing the sub-buffer) or 2 (if it doesn't want control back,
                // and instead allows any wave to take control of the sub-buffer)
                uint8_t flag = __atomic_load_n(&rsrc_.desc_.atomic_flags_[i], __ATOMIC_ACQUIRE);
                if (flag > 1) // buffer is full: process and reset
                {
                    // TODO: process data
                    size_t size = rsrc_.desc_.sub_buffer_sizes_[i];
                    size_t byte_offset = i * rsrc_.desc_.sub_buffer_capacity_;
                    char *message_p = &rsrc_.desc_.buffer_[byte_offset];
                    while (size != 0)
                    {
                        message_t message(message_p);
                        message_handlers_[thread_no].handle(message);
                        assert(message.size() <= size);
                        size -= message.size();
                        message_p += message.size();
                    }

                    rsrc_.desc_.sub_buffer_sizes_[i] = 0;
                    // setting the flag to either 1 (giving contol back to the signaling wave)
                    // or 0 (allowing any wave to take control of the sub-buffer)
                    __atomic_store_n(&rsrc_.desc_.atomic_flags_[i], flag - 2, __ATOMIC_RELEASE);
                }
            }
        }

        // printf("[Host] process_sub_buffers: processing partially full sub-buffers after kernels have finished\n");
        for (size_t i = first; i != last; ++i)
        {
            uint8_t flag = __atomic_load_n(&rsrc_.desc_.atomic_flags_[i], __ATOMIC_ACQUIRE);
            if (flag != 0) // Should not happen, indicates a missing atomic release from device code
            {
                printf("Found non-zero flag for sub-buffer %lu\n", i);
            }
            // TODO: process data
            size_t size = rsrc_.desc_.sub_buffer_sizes_[i];
            size_t byte_offset = i * rsrc_.desc_.sub_buffer_capacity_;
            char *message_p = &rsrc_.desc_.buffer_[byte_offset];
            while (size != 0)
            {
                message_t message(message_p);
                message_handlers_[thread_no].handle(message);
                assert(message.size() <= size);
                size -= message.size();
                message_p += message.size();
            }
        }
    }

} // namespace dh_comms
