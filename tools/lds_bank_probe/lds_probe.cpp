// lds_probe: run one LDS instruction with arbitrary per-lane addresses, one
// dispatch per address pattern, so that hardware counters (rocprofv3, per
// dispatch) show how the LDS services that pattern. See README.md.
//
// Usage: lds_probe <op> <reps> <pattern file>
//        lds_probe --list
// The pattern file has one pattern per line: 64 integers, the byte offset into
// LDS each lane of a single wave uses, or -1 for an inactive lane. Lines that
// are empty or start with '#' are skipped. Each pattern is one dispatch of a
// one-wave kernel in which the active lanes execute <op> <reps> times, each
// followed by s_waitcnt lgkmcnt(0), and nothing else touches LDS.
//
// Ops are written in inline assembly so the exact instruction is known. For
// read2/write2 ops, the lane's two addresses are offset + 4*offset0 and
// offset + 4*offset1 (b32) or 8*offset{0,1} (b64), with the immediates fixed in
// the op name (e.g. read2_b32_o0_1: offset0 0, offset1 1).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

constexpr int WAVE = 64;
constexpr int LDS_BYTES = 65536; // dynamic LDS per dispatch; offsets must stay below this

typedef unsigned v2u __attribute__((ext_vector_type(2)));
typedef unsigned v3u __attribute__((ext_vector_type(3)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));

#define WAIT "\n s_waitcnt lgkmcnt(0)"

// One instruction per op. R(name, type, asm): read into a register of <type>;
// W(name, type, asm): write a value of <type>.
#define OPS(R, W)                                                                     \
  R(read_u8, unsigned, "ds_read_u8 %0, %1")                                           \
  R(read_i8, unsigned, "ds_read_i8 %0, %1")                                           \
  R(read_u16, unsigned, "ds_read_u16 %0, %1")                                         \
  R(read_u8_d16, unsigned, "ds_read_u8_d16 %0, %1")                                   \
  R(read_u8_d16_hi, unsigned, "ds_read_u8_d16_hi %0, %1")                             \
  R(read_u16_d16, unsigned, "ds_read_u16_d16 %0, %1")                                 \
  R(read_u16_d16_hi, unsigned, "ds_read_u16_d16_hi %0, %1")                           \
  R(read_b32, unsigned, "ds_read_b32 %0, %1")                                         \
  R(read_b64, v2u, "ds_read_b64 %0, %1")                                              \
  R(read_b96, v3u, "ds_read_b96 %0, %1")                                              \
  R(read_b128, v4u, "ds_read_b128 %0, %1")                                            \
  R(read2_b32_o0_1, v2u, "ds_read2_b32 %0, %1 offset1:1")                             \
  R(read2_b32_o0_32, v2u, "ds_read2_b32 %0, %1 offset1:32")                           \
  R(read2_b32_o0_64, v2u, "ds_read2_b32 %0, %1 offset1:64")                           \
  R(read2st64_b32_o0_1, v2u, "ds_read2st64_b32 %0, %1 offset1:1")                     \
  R(read2_b64_o0_1, v4u, "ds_read2_b64 %0, %1 offset1:1")                             \
  R(read2_b64_o0_16, v4u, "ds_read2_b64 %0, %1 offset1:16")                           \
  R(read2_b64_o0_32, v4u, "ds_read2_b64 %0, %1 offset1:32")                           \
  R(read2st64_b64_o0_1, v4u, "ds_read2st64_b64 %0, %1 offset1:1")                     \
  W(write_b8, unsigned, "ds_write_b8 %0, %1")                                         \
  W(write_b8_d16_hi, unsigned, "ds_write_b8_d16_hi %0, %1")                           \
  W(write_b16, unsigned, "ds_write_b16 %0, %1")                                       \
  W(write_b16_d16_hi, unsigned, "ds_write_b16_d16_hi %0, %1")                         \
  W(write_b32, unsigned, "ds_write_b32 %0, %1")                                       \
  W(write_b64, v2u, "ds_write_b64 %0, %1")                                            \
  W(write_b96, v3u, "ds_write_b96 %0, %1")                                            \
  W(write_b128, v4u, "ds_write_b128 %0, %1")

// Atomics without return behave like writes (address, data); with return like
// a read that also takes data. A(name, type, asm): returning op.
#define OPS_ATOMIC(W, A)                                                              \
  W(add_u32, unsigned, "ds_add_u32 %0, %1")                                           \
  W(add_u64, v2u, "ds_add_u64 %0, %1")                                                \
  A(add_rtn_u32, unsigned, "ds_add_rtn_u32 %0, %1, %2")                               \
  A(add_rtn_u64, v2u, "ds_add_rtn_u64 %0, %1, %2")

// gfx950 transpose loads: each lane supplies an address, the data returned
// to a lane comes from several lanes' addresses.
#define OPS_TR(R)                                                                     \
  R(read_b64_tr_b4, v2u, "ds_read_b64_tr_b4 %0, %1")                                  \
  R(read_b64_tr_b8, v2u, "ds_read_b64_tr_b8 %0, %1")                                  \
  R(read_b64_tr_b16, v2u, "ds_read_b64_tr_b16 %0, %1")                                \
  R(read_b96_tr_b6, v3u, "ds_read_b96_tr_b6 %0, %1")

// write2 needs two data operands, so it gets its own macro.
#define OPS2(W2)                                                                      \
  W2(write2_b32_o0_1, unsigned, "ds_write2_b32 %0, %1, %2 offset1:1")                 \
  W2(write2_b32_o0_32, unsigned, "ds_write2_b32 %0, %1, %2 offset1:32")               \
  W2(write2_b32_o0_64, unsigned, "ds_write2_b32 %0, %1, %2 offset1:64")               \
  W2(write2st64_b32_o0_1, unsigned, "ds_write2st64_b32 %0, %1, %2 offset1:1")         \
  W2(write2_b64_o0_1, v2u, "ds_write2_b64 %0, %1, %2 offset1:1")                      \
  W2(write2_b64_o0_16, v2u, "ds_write2_b64 %0, %1, %2 offset1:16")                    \
  W2(write2_b64_o0_32, v2u, "ds_write2_b64 %0, %1, %2 offset1:32")                    \
  W2(write2st64_b64_o0_1, v2u, "ds_write2st64_b64 %0, %1, %2 offset1:1")

template <typename T> __device__ unsigned fold(T v) {
  if constexpr (sizeof(T) == 4) return v;
  else {
    unsigned s = 0;
    for (int k = 0; k < int(sizeof(T) / 4); ++k) s += v[k];
    return s;
  }
}
template <typename T> __device__ T splat(unsigned x) {
  if constexpr (sizeof(T) == 4) return x;
  else return T(x);
}

#define KERNEL_R(name, T, insn)                                                       \
  __global__ void k_##name(const int *offs, int reps, unsigned *sink) {               \
    int o = offs[threadIdx.x];                                                        \
    unsigned acc = 0;                                                                 \
    if (o >= 0) {                                                                     \
      for (int r = 0; r < reps; ++r) {                                                \
        T v;                                                                          \
        asm volatile(insn WAIT : "=v"(v) : "v"(o) : "memory");                        \
        acc += fold(v);                                                               \
      }                                                                               \
    }                                                                                 \
    sink[threadIdx.x] = acc;                                                          \
  }
#define KERNEL_W(name, T, insn)                                                       \
  __global__ void k_##name(const int *offs, int reps, unsigned *sink) {               \
    int o = offs[threadIdx.x];                                                        \
    if (o >= 0) {                                                                     \
      T v = splat<T>(threadIdx.x);                                                    \
      for (int r = 0; r < reps; ++r)                                                  \
        asm volatile(insn WAIT ::"v"(o), "v"(v) : "memory");                          \
    }                                                                                 \
    sink[threadIdx.x] = 0;                                                            \
  }
#define KERNEL_W2(name, T, insn)                                                      \
  __global__ void k_##name(const int *offs, int reps, unsigned *sink) {               \
    int o = offs[threadIdx.x];                                                        \
    if (o >= 0) {                                                                     \
      T v = splat<T>(threadIdx.x), w = splat<T>(threadIdx.x + 1);                     \
      for (int r = 0; r < reps; ++r)                                                  \
        asm volatile(insn WAIT ::"v"(o), "v"(v), "v"(w) : "memory");                  \
    }                                                                                 \
    sink[threadIdx.x] = 0;                                                            \
  }
#define KERNEL_A(name, T, insn)                                                       \
  __global__ void k_##name(const int *offs, int reps, unsigned *sink) {               \
    int o = offs[threadIdx.x];                                                        \
    unsigned acc = 0;                                                                 \
    if (o >= 0) {                                                                     \
      T w = splat<T>(threadIdx.x);                                                    \
      for (int r = 0; r < reps; ++r) {                                                \
        T v;                                                                          \
        asm volatile(insn WAIT : "=v"(v) : "v"(o), "v"(w) : "memory");                \
        acc += fold(v);                                                               \
      }                                                                               \
    }                                                                                 \
    sink[threadIdx.x] = acc;                                                          \
  }
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
#define KERNEL_TR KERNEL_R
#else // not encodable before gfx950: the kernel exists but traps
#define KERNEL_TR(name, T, insn)                                                      \
  __global__ void k_##name(const int *, int, unsigned *) { __builtin_trap(); }
#endif
OPS(KERNEL_R, KERNEL_W)
OPS2(KERNEL_W2)
OPS_ATOMIC(KERNEL_W, KERNEL_A)
OPS_TR(KERNEL_TR)

typedef void (*kernel_t)(const int *, int, unsigned *);
struct op_t {
  const char *name;
  kernel_t k;
  bool gfx950_only;
};
#define ENTRY(name, T, insn) {#name, k_##name, false},
#define ENTRY_TR(name, T, insn) {#name, k_##name, true},
static const op_t ops[] = {OPS(ENTRY, ENTRY) OPS2(ENTRY) OPS_ATOMIC(ENTRY, ENTRY) OPS_TR(ENTRY_TR)};

static void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    exit(1);
  }
}

int main(int argc, char **argv) {
  hipDeviceProp_t prop;
  check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  bool gfx950 = !strncmp(prop.gcnArchName, "gfx950", 6);
  if (argc == 2 && !strcmp(argv[1], "--list")) { // ops this GPU can run
    for (const op_t &op : ops)
      if (gfx950 || !op.gfx950_only) printf("%s\n", op.name);
    return 0;
  }
  if (argc != 4) {
    fprintf(stderr, "usage: %s <op> <reps> <pattern file> | --list\n", argv[0]);
    return 2;
  }
  const op_t *op = nullptr;
  for (const op_t &o : ops)
    if (!strcmp(o.name, argv[1])) op = &o;
  if (!op || (op->gfx950_only && !gfx950)) {
    fprintf(stderr, "unknown op %s on %s (see --list)\n", argv[1], prop.gcnArchName);
    return 2;
  }
  int reps = atoi(argv[2]);
  std::ifstream in(argv[3]);
  if (!in) {
    fprintf(stderr, "cannot read %s\n", argv[3]);
    return 2;
  }
  std::vector<std::vector<int>> patterns;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::vector<int> p;
    int v;
    while ((int)p.size() < WAVE && ls >> v) p.push_back(v);
    if ((int)p.size() != WAVE) {
      fprintf(stderr, "pattern %zu: need %d offsets\n", patterns.size(), WAVE);
      return 2;
    }
    for (int x : p)
      if (x >= LDS_BYTES - 1024) { // read2/write2 immediates reach up to 64*8*... keep headroom
        fprintf(stderr, "pattern %zu: offset %d too large\n", patterns.size(), x);
        return 2;
      }
    patterns.push_back(p);
  }
  int *d_offs;
  unsigned *d_sink;
  check(hipMalloc(&d_offs, WAVE * sizeof(int)), "hipMalloc");
  check(hipMalloc(&d_sink, WAVE * sizeof(unsigned)), "hipMalloc");
  for (const auto &p : patterns) {
    check(hipMemcpy(d_offs, p.data(), WAVE * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy");
    hipLaunchKernelGGL(op->k, dim3(1), dim3(WAVE), LDS_BYTES, 0, d_offs, reps, d_sink);
    check(hipDeviceSynchronize(), "dispatch");
  }
  printf("%s: %zu dispatches\n", op->name, patterns.size());
  return 0;
}
