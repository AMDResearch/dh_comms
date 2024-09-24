#pragma once

#include <map>
#include "message_processor_base.h"
namespace dh_comms
{

    class memory_heatmap_t : public message_processor_base
    {
    public:
        memory_heatmap_t(size_t page_size, bool verbose = false);
        virtual ~memory_heatmap_t(){}
        size_t operator()(char *&message_p, size_t size, size_t sub_buf_no);
        void show() const;

    private:
        bool verbose_;
        size_t page_size_;
        std::map<uint64_t, size_t> page_counts_;
    };

} // namespace dh_comms