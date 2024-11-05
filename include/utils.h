#pragma once

#include <string>

inline std::string exec2binstr(uint64_t exec) {
  std::string bits;
  uint64_t mask = 1UL << 63;
  for (size_t i = 0; i != 4; ++i) {
    for (size_t j = 0; j != 4; ++j) {
      for (size_t k = 0; k != 4; ++k) {
        bits += exec & mask ? "1" : "0";
        mask >>= 1;
      }
      if (j < 3) {
        bits += " ";
      }
    }
    if (i < 3) {
      bits += " | ";
    }
  }
  return bits;
}