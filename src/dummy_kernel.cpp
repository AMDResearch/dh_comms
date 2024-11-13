#include "dh_comms.h"
#include "dh_comms_dev.h"

// Purpose of this dummy kernel is to make sure that device functions are included in libdh_comms.so
// This seems to be needed under ROCM 6.2.1, while it wasn't under older ROCm versions (e.g. 6.0.2)

namespace dh_comms {
__global__ void dummy() {}
} // namespace dh_comms
