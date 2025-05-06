// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "memory_heatmap.h"

#include "data_headers.h"
#include "message.h"

#include <cassert>
#include <cstdio>
#include <vector>

namespace dh_comms {
memory_heatmap_t::memory_heatmap_t(size_t page_size, bool verbose)
    : verbose_(verbose),
      page_size_(page_size) {}

bool memory_heatmap_t::handle(const message_t &message, const std::string& kernel_name, kernelDB::kernelDB& kdb) {
    // This if block is just to get the compiler to quick throwing errors for unused parameters
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    return handle(message);
}
bool memory_heatmap_t::handle(const message_t &message) {
  if (message.wave_header().user_type != message_type::address) {
    if (verbose_) {
      printf("memory_heatmap: skipping message with user type 0x%x\n", message.wave_header().user_type);
    }
    return false;
  }
  assert(message.data_item_size() == sizeof(uint64_t));
  for (size_t i = 0; i != message.no_data_items(); ++i) {
    uint64_t address = *(const uint64_t *)message.data_item(i);
    // map address to lowest address in page and update page count
    address /= page_size_;
    address *= page_size_;
    ++page_counts_[address];
    if (verbose_) {
      printf("memory_heatmap: added address 0x%lx to map\n", address);
    }
  }
  return true;
}

void memory_heatmap_t::report(const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    report();
}

void memory_heatmap_t::report() {
  if (page_counts_.size() != 0) {
    printf("memory heatmap report:\n\tpage size = %lu\n", page_size_);
  }
  for (const auto &[first_page_address, count] : page_counts_) {
    auto last_page_address = first_page_address + page_size_ - 1;
    printf("\tpage [%016lx:%016lx] %12lu accesses\n", first_page_address, last_page_address, count);
  }
}

void memory_heatmap_t::clear() { page_counts_.clear(); }

} // namespace dh_comms
