#pragma once

namespace dh_comms
{
    //! \brief Messages start with a wave header containing information that partains to the whole wave.
    //!
    //! User device code does not use wave headers directly, but user host code may.
    struct wave_header_t
    {
        uint64_t exec; //!< Execution mask of the wavefront submitting the message.

        uint64_t data_size;             //!< \brief Size of the data following the wave header.
                                        //!<
                                        //!< This is computed as: (number of active lanes) * (lane header (4 bytes) +
                                        //!< number of data bytes per lane), rounded up to nearest multiple of 4 bytes.
        uint64_t timestamp;             //!< GPU timestamp of the moment the message was submitted.
        uint32_t active_lane_count : 7; //!< number of active lanes in the wavefront, i.e., number of 1-bits in the execution mask
        uint32_t src_loc_idx : 25;      //!< Integer that can be used to uniquely identify the source code location from which
                                        //!< the message was submitted.
        uint32_t user_type;             //!< \brief User-defined tag that indicates the content/interpretation of the data.
                                        //!<
                                        //!< Since kernels can submit any type of data that is valid on the device, such as
                                        //!< an uint64_t (which could represent a memory address or a loop counter) or a
                                        //!< struct containing several basic type), host code that processes the messages must
                                        //!< be able to determine what kind of data is in the message, and how to process it.
                                        //!< Note that the processing code on the host is to be provided by the user, although
                                        //!< some basic data processor classes are included with dh_comms. By using the user_type
                                        //!< tag, kernel code can indicate the type of data in the message, so that the host
                                        //!< processing code knows what to do with it. The user_type tag could also be used to
                                        //!< distinguish between messages from different kernels or kernel dispatches, if we are
                                        //!< using a single dh_comms object to pass messages from the device to the host, as
                                        //!< opposed to having a separate dh_comms object per kernel or kernel dispatch. Another
                                        //!< use of the user_type tag is to distinguish between memory reads vs writes, if our
                                        //!< messages contain the memory addresses (say, if we are building a memory heatmap of
                                        //!< the application), or to distinguish between identical message types submitted from
                                        //!< different locations in the ISA or source code.
        uint16_t block_idx_x;           //!< blockIdx.x value of the workgroup to which the wave belongs.
        uint16_t block_idx_y;           //!< blockIdx.y value of the workgroup to which the wave belongs.
        uint16_t block_idx_z;           //!< blockIdx.z value of the workgroup to which the wave belongs.
        uint16_t wave_num : 4;          //!< Wave number withing workgroup; there can be at most 16 waves per workgroup.
        uint16_t xcc_id : 4;            //!< \brief Number of the XCC on which the wavefront runs; zero for pre-MI300 hardware
                                        //!<
                                        //!< From MI300 on, CDNA hardware is partitioned into multiple XCCs (these are the individual
                                        //!< dies). MI200 has two dies, but these are called XCDs, and opposed to MI300, the MI200
                                        //!< dies are individual devices; the workgroups of a kernel can only run on one of them.
                                        //!< MI300 can be booted into various partitioning schemes, including SPX, where all the dies
                                        //!< combine into a single devices, and the waves of a kernel can be distributed over all
                                        //!< XCCs. Regardless of the CDNA generation, the dies are subdivided into SEs (Shader Engines),
                                        //!< and there can be up to 8 SEs per die. SEs are subdivided into CUs (Compute Units), of which
                                        //!< there can be up to 16 per SE. If as SE has fewer than 16 CUs, they need not be numbered
                                        //!< consecutively, nor does the lowest CU number need to be 0. In fact, the CU numbers may
                                        //!< be different from one SE on the device to the next SE.
        uint16_t se_id : 3;             //!< Number of the SE of the XCC on which the wavefront runs.
        uint16_t cu_id : 4;             //!< Number of the CU of the SE on which the wavefront runs.
        uint16_t unused16 : 1;          //!< Padding; reserved for future use.

        //! Wave header constructor; wave header members for which there is no corresponding
        //! constructor argument are detected and assigned by the constructor.
        __device__ wave_header_t(uint64_t exec, uint64_t data_size, uint64_t timestamp, uint32_t active_lane_count,
                                 uint32_t src_loc_idx, uint32_t user_type);
    };

    //! \brief After the wave header, there's a lane header for each of the active lanes in the wavefront.
    //!
    //! The lane header contains the threadIdx.[x,y,z] values for the lane. Since the thread indices range from
    //! 0 to at most 1023 in each dimension, they can be packed into a 4-byte struct.
    //! User device code does not use lane headers directly, but user host code may.

    struct lane_header_t
    {
        uint32_t thread_idx_x : 10;
        uint32_t thread_idx_y : 10;
        uint32_t thread_idx_z : 10;
        uint32_t unused : 2;

        //! The lane header constructor fills the thread_idx_[x,y,z] fields.
        __device__ lane_header_t();
    };

} // namespace dh_comms