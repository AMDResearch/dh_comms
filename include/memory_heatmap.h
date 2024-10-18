#pragma once

#include <map>
#include "message_handlers.h"
namespace dh_comms
{

    class memory_heatmap_t : public message_handler_base
    {
    public:
        memory_heatmap_t(size_t page_size = 1024 * 1024, bool verbose = false);
        memory_heatmap_t(const memory_heatmap_t&) = default;
        virtual ~memory_heatmap_t(){};
        virtual bool handle(const message_t &message) override;
        virtual void report() override;
        virtual void clear() override;

    private:
        bool verbose_;
        size_t page_size_;
        std::map<uint64_t, size_t> page_counts_;
    };

} // namespace dh_comms