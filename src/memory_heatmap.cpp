#include <cstdio>
#include <vector>
#include "memory_heatmap.h"
#include "data_headers.h"

namespace dh_comms
{
    memory_heatmap_t::memory_heatmap_t(size_t page_size, bool verbose)
        : verbose_(verbose),
          page_size_(page_size)
    {
    }

    size_t memory_heatmap_t::operator()(char *&message_p, size_t size, size_t sub_buf_no)
    {
        // process information in the wave header
        wave_header_t *wave_header_p = (wave_header_t *)message_p;
        size_t data_size = wave_header_p->data_size;
        uint32_t active_lane_count = wave_header_p->active_lane_count;
        if (verbose_)
        {
            printf("[Host] %zu bytes of data remaining in sub-buffer %zu\n", size, sub_buf_no);
            printf("wave_header:\n");
            printf("\texec = 0x%016lx\n", wave_header_p->exec);
            printf("\tactive_lane_count = %u\n", active_lane_count);
            printf("\t[block]:wave = [%u,%u,%u]:%u\n", wave_header_p->block_idx_x, wave_header_p->block_idx_y,
                   wave_header_p->block_idx_z, wave_header_p->wave_num);
            printf("\txcc:se:cu = %02u:%02u:%02u\n", wave_header_p->xcc_id, wave_header_p->se_id,
                   wave_header_p->cu_id);
        }
        message_p += sizeof(wave_header_t);

        // process information in the lane headers
        if(verbose_)
        {
            printf("lane headers:\n");
        }
        lane_header_t *lane_header_p = (lane_header_t *)message_p;
        for (uint32_t lane = 0; lane != active_lane_count; ++lane)
        {
            if (verbose_)
            {
                printf("\t[thread] = [%u,%u,%u]\n", lane_header_p->thread_idx_x,
                       lane_header_p->thread_idx_y, lane_header_p->thread_idx_z);
            }
            ++lane_header_p;
        }

        // process the 64-bit addresses; they are split into 32-bit dwords;
        if(verbose_)
        {
            printf("data:\n");
        }
        std::vector<uint64_t> addresses(active_lane_count);
        uint32_t* dword_p = (uint32_t*)lane_header_p;
        // first, get the lower-order bits
        for(size_t lane=0; lane != active_lane_count; ++lane)
        {
            addresses[lane] = *dword_p;
            if(verbose_){
                printf("\taddress lo: %lu\n", addresses[lane]);
            }
            ++dword_p;
        }

        // next, get the higher-order bits
        for(size_t lane=0; lane != active_lane_count; ++lane)
        {
            uint64_t address_hi = *dword_p;
            if(verbose_){
                printf("\taddress hi: %lu\n", address_hi);
            }
            addresses[lane] |= (address_hi << 32);
            ++dword_p;
        }
        if(verbose_){
            for(size_t lane=0; lane != active_lane_count; ++lane)
            {
                printf("\tfull address: %lu\n", addresses[lane]);
            }
        }

        // update page counts with the addresses observed
        for(auto address: addresses)
        {
            // map address to lowest address in page
            address /= page_size_;
            address *= page_size_;
            ++page_counts_[address];
        }

        message_p += data_size;
        size -= (sizeof(wave_header_t) + data_size);
        if (verbose_)
        {
            printf("\n");
        }

        return size;
    }

    void memory_heatmap_t::show() const
    {
        printf("memory heatmap: page size = %lu\n", page_size_);
        for( const auto& [first_page_address, count] : page_counts_)
        {
            auto last_page_address = first_page_address + page_size_ -1;
            printf("page [%016lx:%016lx] %12lu accesses\n", first_page_address, last_page_address, count);
        }
    }

} // namespace dh_comms