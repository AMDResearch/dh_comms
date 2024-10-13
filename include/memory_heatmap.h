#pragma once

#include <map>
#include "message_processor_base.h"
#include "message_handlers.h"
namespace dh_comms
{

    class memory_heatmap_t : public message_processor_base
    {
    public:
        memory_heatmap_t(size_t page_size, bool verbose = false);
        virtual ~memory_heatmap_t() {}
        virtual size_t operator()(char *&message_p, size_t size, size_t sub_buf_no);
        void show() const;

    private:
        bool verbose_;
        size_t page_size_;
        std::map<uint64_t, size_t> page_counts_;
    };

    class memory_heatmap_v2_t : public message_handler_base
    {
    public:
        memory_heatmap_v2_t(size_t page_size = 4096, bool verbose = false);
        virtual ~memory_heatmap_v2_t();
        virtual bool handle(const message_t &message) override;
        virtual void merge_state(message_handler_base &other) override;
        void show() const;

    private:
        bool verbose_;
        size_t page_size_;
        std::map<uint64_t, size_t> page_counts_;
    };

} // namespace dh_comms