#pragma once
#include "data_headers.h"
#include "message_handlers.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace dh_comms {
class dh_comms_mem_mgr {
public:
  dh_comms_mem_mgr();
  virtual ~dh_comms_mem_mgr();
  virtual void *calloc(std::size_t size);
  virtual void *calloc_device_memory(std::size_t size);
  virtual void free(void *);
  virtual void free_device_memory(void *);
  virtual void *copy(void *dst, void *src, std::size_t size);
  virtual void *copy_to_device(void *dst, const void *src, std::size_t size);
  virtual void zero(void *buffer, std::size_t size);
  virtual void zero_device_memory(void *buffer, std::size_t size);
};

struct dh_comms_descriptor {
  std::size_t no_sub_buffers_;      //!< Number of sub-buffers into which the main data buffer is partitioned.
  std::size_t sub_buffer_capacity_; //!< The maximum number of bytes each of the sub-buffers can hold.
  char *buffer_;                    //!< Pointer to the main data buffer.
  size_t *sub_buffer_sizes_;        //!< Pointer to an array of no_sub_buffers_ entries. Each entry holds the number
                                    //!< of bytes (0 <= size <= sub_buffer_capacity_) currently in the corresponding
                                    //!< sub-buffer.
  uint32_t *error_bits_;            //!< Pointer to an array of no_sub_buffers_ entries, one for each sub-buffer,
                                    //!< used to track error conditions for the sub-buffer, such as a wave attempting
                                    //!< to write more than sub_buffer_capacity_ bytes to a sub-buffer.
  uint8_t *atomic_flags_d_;         //!< Used for synchronization between different waves in device code.
  uint8_t *atomic_flags_hd_;        //!< Used for synchronization between host and device code
};
//! \brief Keeps track of resources used by device and host code for exchanging data.
//!
//! The main data buffer is partitioned into a number of equal-sized sub-buffers to
//! allow for concurrent access. Multiple waves can write to different sub-buffers
//! simultaneously. Waves that want to write to the same sub-buffer are serialized
//! using atomics.
struct dh_comms_resources {
  dh_comms_descriptor desc_;
  dh_comms_mem_mgr &mgr_;

  //! Constructor; allocates memory resources based on its arguments.
  dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity, dh_comms_mem_mgr &mgr);
  dh_comms_resources(const dh_comms_resources &) = delete;
  dh_comms_resources &operator=(const dh_comms_resources &) = delete;
  //! Destructor; releases allocated memory.
  ~dh_comms_resources();
};

//! \brief Orchestrates allocation of resources for message passing from device to host code
//! and processing of messages on the host
class dh_comms {
public:
  dh_comms(std::size_t no_sub_buffers,      //!< Number of sub-buffers into which the main data buffer is partitioned.
           std::size_t sub_buffer_capacity, //!< The maximum number of bytes each of the sub-buffers can hold.
           bool verbose = false,            //!< Controls how chatty the code is.
           bool install_default_handlers = false, dh_comms_mem_mgr *mgr = NULL, bool handlers_pass_through = true);
  ~dh_comms();
  dh_comms(const dh_comms &) = delete;
  dh_comms &operator=(const dh_comms &) = delete;
  dh_comms_descriptor *get_dev_rsrc_ptr(); //!< Returns a pointer to a dh_comms_resources struct in device memory.

  void start(); //!< Start the message processing threads on the host.
  void stop();  //!< \brief Stop message processing on the host.
                //!<
                //!< It is the responsibility
                //!< of calling code to make sure kernels have finished by e.g. issuing
                //!< a hipDeviceSyncronize() or other synchronization call.
  void append_handler(std::unique_ptr<message_handler_base> &&message_handler);
  //!< \brief Append a message handler to the end of the handler chain.
  //!<
  //!< A handler may handle just a single message type, or it may handle
  //!< multiple message types. The first handler that can handle a message
  //!< of a particular type gets to handle it. If a message cannot be
  //!< handled by any handler in the chain, it is silently dropped.
  void clear_handler_states();                //!< Keep the message handlers, but clear their states, so that they
                                              //!< can be reused for a subsequent run.
  void delete_handlers();                     //!< delete the message handlers, so that a new set can be installed
                                              //!< for a subsequent run.
  void report(bool auto_clear_states = true); //!< \brief calls the report() function of all message handlers
                                              //!<
                                              //!< if auto_clear_states is true, the states of the message handlers
                                              //!< will be cleared after reporting

private:
  void process_sub_buffers();
  void processing_loop(bool is_final_loop);
  void install_default_message_handlers();

private:
  dh_comms_mem_mgr default_mgr_;
  dh_comms_mem_mgr *mgr_;
  dh_comms_resources rsrc_;
  dh_comms_descriptor *dev_rsrc_p_;
  volatile bool running_;
  const bool verbose_;
  message_handler_chain_t message_handler_chain_;
  std::thread sub_buffer_processor_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::chrono::time_point<std::chrono::steady_clock> stop_time_;
  std::size_t bytes_processed_;
};

} // namespace dh_comms
