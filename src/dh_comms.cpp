#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

#include <hip/hip_runtime.h>
#include "hip_utils.h"

#include "dh_comms.h"
#include "data_headers.h"

namespace dh_comms
{

    dh_comms_mem_mgr::dh_comms_mem_mgr()
    {
        return;
    }

    dh_comms_mem_mgr::~dh_comms_mem_mgr()
    {
    }

    void * dh_comms_mem_mgr::alloc(std::size_t size)
    {
        void *buffer;
        CHK_HIP_ERR(hipHostMalloc(&buffer, size, hipHostMallocCoherent));
        zero((char *)buffer, size);
        return buffer;
    }

    void dh_comms_mem_mgr::free(void *ptr)
    {
        CHK_HIP_ERR(hipFree(ptr));
        return;
    }

    void * dh_comms_mem_mgr::copy(void *dst, void *src, std::size_t size)
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
    T *clone_to_device(const T &host_data)
    {
        T *device_data;
        CHK_HIP_ERR(hipMalloc(&device_data, sizeof(T)));
        CHK_HIP_ERR(hipMemcpy(device_data, &host_data, sizeof(T), hipMemcpyHostToDevice));
        return device_data;
    }

} // unnamed namespace

namespace dh_comms
{
    dh_comms_resources::dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity, dh_comms_mem_mgr& mgr)
        : mgr_(mgr),
          no_sub_buffers_(no_sub_buffers),
          sub_buffer_capacity_(sub_buffer_capacity),
          buffer_((decltype(buffer_))mgr_.alloc(no_sub_buffers_ * sub_buffer_capacity_)),
          sub_buffer_sizes_((decltype(sub_buffer_sizes_))mgr_.alloc(no_sub_buffers_ * sizeof(decltype(*sub_buffer_sizes_)))),
          error_bits_((decltype(error_bits_))mgr_.alloc(sizeof(decltype(*error_bits_)))),
          atomic_flags_((decltype(atomic_flags_))mgr_.alloc(no_sub_buffers_ * sizeof(decltype(*atomic_flags_))))
          //buffer_((decltype(buffer_))allocate_shared_buffer(no_sub_buffers_ * sub_buffer_capacity_)),
          //sub_buffer_sizes_((decltype(sub_buffer_sizes_))allocate_shared_buffer(no_sub_buffers_ * sizeof(decltype(*sub_buffer_sizes_)))),
          //error_bits_((decltype(error_bits_))allocate_shared_buffer(sizeof(decltype(*error_bits_)))),
          //atomic_flags_((decltype(atomic_flags_))allocate_shared_buffer(no_sub_buffers_ * sizeof(decltype(*atomic_flags_))))
    {
    }

    dh_comms_resources::~dh_comms_resources()
    {
        mgr_.free(atomic_flags_);
        mgr_.free(error_bits_);
        mgr_.free(sub_buffer_sizes_);
        mgr_.free(buffer_);
        //CHK_HIP_ERR(hipFree(atomic_flags_));
        //CHK_HIP_ERR(hipFree(error_bits_));
        //CHK_HIP_ERR(hipFree(sub_buffer_sizes_));
        //CHK_HIP_ERR(hipFree(buffer_));
    }

    dh_comms::dh_comms(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity,
                       message_processor_base &message_processor,
                       std::size_t no_host_threads, bool verbose, dh_comms_mem_mgr *mgr)
        : mgr_( mgr ? mgr : &default_mgr_),
          rsrc_(no_sub_buffers, sub_buffer_capacity, *mgr_),
          dev_rsrc_p_(clone_to_device(rsrc_)),
          sub_buffer_processors_(init_host_threads(no_host_threads, message_processor.is_thread_safe())),
          message_processor_(message_processor),
          teardown_(false),
          verbose_(verbose)
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
        if (*rsrc_.error_bits_ & 1)
        {
            printf("Error detected: data from device dropped because message size was larger than sub-buffer size\n");
        }
        CHK_HIP_ERR(hipFree(dev_rsrc_p_));
    }

    dh_comms_resources *dh_comms::get_dev_rsrc_ptr()
    {
        return dev_rsrc_p_;
    }

    std::vector<std::thread> dh_comms::init_host_threads(std::size_t no_host_threads,
                                                         bool message_processor_is_thread_safe)
    {
        assert(no_host_threads != 0);
        if (no_host_threads > 1 and not message_processor_is_thread_safe)
        {
            printf("Thread-safe message processor required for multi-threaded host processing; exiting\n");
            exit(EXIT_FAILURE);
        }
        std::size_t no_sub_buffers_per_thread = rsrc_.no_sub_buffers_ / no_host_threads;
        std::size_t remainder = rsrc_.no_sub_buffers_ % no_host_threads;

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
            sub_buffer_processors.emplace_back(std::thread(&dh_comms::process_sub_buffers, this, first, last));
            first = last;
        }
        assert(last == rsrc_.no_sub_buffers_);

        return sub_buffer_processors;
    }

    void dh_comms::process_sub_buffers(std::size_t first, std::size_t last)
    {
        while (not teardown_)
        {
            for (size_t i = first; i != last; ++i)
            {
                // when the sub-buffer for a wave on the device is full, it will
                // set the flag to either 3 (if it wants control back after the host
                // is done processing the sub-buffer) or 2 (if it doesn't want control back,
                // and instead allows any wave to take control of the sub-buffer)
                uint8_t flag = __atomic_load_n(&rsrc_.atomic_flags_[i], __ATOMIC_ACQUIRE);
                if (flag > 1) // buffer is full: process and reset
                {
                    // TODO: process data
                    size_t size = rsrc_.sub_buffer_sizes_[i];
                    size_t byte_offset = i * rsrc_.sub_buffer_capacity_;
                    char *message_p = &rsrc_.buffer_[byte_offset];
                    while (size != 0)
                    {
                        size = message_processor_(message_p, size, i);
                    }

                    rsrc_.sub_buffer_sizes_[i] = 0;
                    // setting the flag to either 1 (giving contol back to the signaling wave)
                    // or 0 (allowing any wave to take control of the sub-buffer)
                    __atomic_store_n(&rsrc_.atomic_flags_[i], flag - 2, __ATOMIC_RELEASE);
                }
            }
        }

        // printf("[Host] process_sub_buffers: processing partially full sub-buffers after kernels have finished\n");

        for (size_t i = first; i != last; ++i)
        {
            uint8_t flag = __atomic_load_n(&rsrc_.atomic_flags_[i], __ATOMIC_ACQUIRE);
            if (flag != 0) // Should not happen, indicates a missing atomic release from device code
            {
                printf("Found non-zero flag for sub-buffer %lu\n", i);
            }
            // TODO: process data
            size_t size = rsrc_.sub_buffer_sizes_[i];
            size_t byte_offset = i * rsrc_.sub_buffer_capacity_;
            char *message_p = &rsrc_.buffer_[byte_offset];
            while (size != 0)
            {
                size = message_processor_(message_p, size, i);
            }
        }
    }

} // namespace dh_comms
