#include <cstdio>
#include <vector>
#include <cassert>
#include "memory_heatmap.h"
#include "data_headers.h"
#include "message.h"

namespace dh_comms
{
    memory_heatmap_t::memory_heatmap_t(size_t page_size, bool verbose)
        : verbose_(verbose),
          page_size_(page_size)

    {
        (void)verbose_;
        (void)page_size_;
    }

    memory_heatmap_t::~memory_heatmap_t()
    {
        show();
    }

    bool memory_heatmap_t::handle(const message_t &message)
    {
        if ((e_message)message.wave_header.user_type != e_message::address)
        {
            return false;
        }
        for (const auto &charv : message.data)
        {
            assert(charv.size() == 8);
            uint64_t address = *(uint64_t *)charv.data();
            // map address to lowest address in page and update page count
            address /= page_size_;
            address *= page_size_;
            ++page_counts_[address];
        }
        return true;
    }

    void memory_heatmap_t::merge_state(message_handler_base &other)
    {
        memory_heatmap_t &other_mh = dynamic_cast<memory_heatmap_t &>(other);
        for (const auto &[page, count] : other_mh.page_counts_)
        {
            page_counts_[page] += count;
        }

        other_mh.page_counts_.clear();
        return;
    }

    void memory_heatmap_t::show() const
    {
        printf("memory heatmap: page size = %lu\n", page_size_);
        for (const auto &[first_page_address, count] : page_counts_)
        {
            auto last_page_address = first_page_address + page_size_ - 1;
            printf("page [%016lx:%016lx] %12lu accesses\n", first_page_address, last_page_address, count);
        }
    }

} // namespace dh_comms