# dh_comms: device to host message passing

## Quickstart
```
# configure and build
mkdir build
cd build
cmake ..
make
# run example
examples/bin/heatmap_example
```

## Design and Usage

Objects of type `dh_comms` manage memory buffers that can be accessed from both device code and host code.
The main buffer, used for submitting messages from device code, is partitioned into a number of sub-buffers
of a given size (both the number of sub-buffers and the size of the sub-buffers are arguments of the `dh_comms`
constructor). A wave on the device writes to sub-buffer _i_, where _i_ is computed as the flattened workgroup id
of the wave modulo the number of sub-buffers. This balances the number of waves that write to each of the sub-buffers.
If a wave wants to submit a message to a sub-buffer but finds that there is insufficient space, it signals the host
code to empty the buffer, and once the host signals the wave that the buffer has been emptied, the wave proceeds
with submitting its message.

Apart from the main data buffer, `dh_comms` uses additional memory buffers accessible to both device and host code.
One of these additional buffers keeps track of the sizes of the (main) sub-buffers (i.e., not the fixed capacity of
the sub-buffers, but the number of bytes written to the sub-buffers by various waves, prior to the host code processing
the data). Another additional buffer is used as an array of atomic flags, one per (main) sub-buffer, to control
exclusive access by device and host code, to let waves signal host code when a sub-buffer is full, and to let
host code signal device code when the data in the sub-buffer has been processed. Pointers to the main and additional
buffers are combined into a struct of type `dh_comms_resources`.

The `dh_comms` constructor takes care of allocating memory for the main and additional memory buffers (currently in
host-pinned memory), filling in a `dh_comms_resources` object with pointers to the buffers, and copying the
`dh_comms_resources` object to device memory. A pointer to the `dh_comms_resource` object in device memory can
be retrieved with the function `dh_comms::get_dev_rsrc_ptr()`, and this device pointer can then be passed to kernels
from which we want to pass messages.

To submit a message from a kernel, the kernel uses the function `dh_comms::v_submit_message()`, which takes three arguments.
The first argument is a pointer to the `dh_comms_resources` struct on the device that we passed to the kernel. The second
argument is a message of any type `T` that is valid in device code, e.g., a `uint64_t` if we want to pass memory addresses
to the host, any integer type if we want to pass e.g. loop counts in device code, or any user-defined struct containing
any set of basic type. Finally, the third argument is an integer to tag the type of data passed in the message.

The purpose of the third argument is the following: `dh_comms` code does not restrict user code with respect to what
kind of data can be passed from device code to host code, and therefore, it doesn't know how to process the data on the
host. Host code polls the memory buffers and can iterate over the sub-buffers once they are full, but the processing
that is done in each iteration is done by a callback to user-provided code (although `dh_comms` does provide some
types of callbacks that can be used out-of-the-box, such as construction a memory access map, or writing the data to disk
for further post-processing). This is where the third argument of `dh_comms::v_submit_message()` comes in: the integer tag
is written into a message header so that the processing call-back code can decide how to process it, in case multiple
types of messages are submitted from device code.