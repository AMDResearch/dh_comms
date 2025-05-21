#include "json_output.h"

namespace dh_comms {

// Define static members
JsonOutputManager* JsonOutputManager::instance_ = nullptr;
std::mutex JsonOutputManager::instance_mutex_;

} // namespace dh_comms 