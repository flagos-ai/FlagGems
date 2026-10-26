#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
__device__ __forceinline__ bool normalize_index(long long raw, long long dim, long long* normalized) {
  // Check before adding dim so LLONG_MIN cannot overflow into a valid index.
  if (raw < 0) {
    if (raw < -dim) return false;
    raw += dim;
  } else if (raw >= dim) {
    return false;
  }
  *normalized = raw;
  return true;
}
__global__ void scatter_u16(unsigned short* x,const long long* idx,const unsigned short* v,int M,int rows,int N,int broadcast){int j=blockIdx.x*blockDim.x+threadIdx.x;int i=blockIdx.y;if(i<M&&j<N){long long r;if(normalize_index(idx[i],rows,&r))x[r*N+j]=v[broadcast?j:i*N+j];}}
extern "C" void launch_scatter(unsigned short* x,const long long* idx,const unsigned short* v,int M,int rows,int N,int broadcast,musaStream_t s){scatter_u16<<<dim3((N+255)/256,M),dim3(256),0,s>>>(x,idx,v,M,rows,N,broadcast);}
__global__ void scatter_f32(float* x,const long long* idx,const float* v,int M,int rows,int N,int broadcast){int j=blockIdx.x*blockDim.x+threadIdx.x;int i=blockIdx.y;if(i<M&&j<N){long long r;if(normalize_index(idx[i],rows,&r))x[r*N+j]=v[broadcast?j:i*N+j];}}
extern "C" void launch_scatter_f32(float* x,const long long* idx,const float* v,int M,int rows,int N,int broadcast,musaStream_t s){scatter_f32<<<dim3((N+255)/256,M),dim3(256),0,s>>>(x,idx,v,M,rows,N,broadcast);}

template <typename T>
__global__ void row_clone_scatter(T* output, const T* input,
                                  const long long* indices, const T* values,
                                  int update_count, int rows, int columns,
                                  bool broadcast_values) {
  __shared__ int selected_update;
  int row = blockIdx.y;
  if (threadIdx.x == 0) {
    selected_update = -1;
    for (int update = 0; update < update_count; ++update) {
      long long normalized;
      if (normalize_index(indices[update], rows, &normalized) && normalized == row)
        selected_update = update;
    }
  }
  __syncthreads();
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column < columns) {
    long long offset = (long long)row * columns + column;
    if (selected_update < 0) output[offset] = input[offset];
    else output[offset] = values[broadcast_values ? column :
                                 (long long)selected_update * columns + column];
  }
}

template <typename T>
static void launch_row_clone_scatter_impl(T* output, const T* input,
                                          const long long* indices,
                                          const T* values, int update_count,
                                          int rows, int columns,
                                          bool broadcast_values,
                                          musaStream_t stream) {
  dim3 grid((columns + 255) / 256, rows);
  row_clone_scatter<T><<<grid, 256, 0, stream>>>(output, input, indices, values,
      update_count, rows, columns, broadcast_values);
}

extern "C" void launch_row_clone_scatter_f16(unsigned short* output,
    const unsigned short* input, const long long* indices,
    const unsigned short* values, int update_count, int rows, int columns,
    bool broadcast_values, musaStream_t stream) {
  launch_row_clone_scatter_impl(reinterpret_cast<__half*>(output),
      reinterpret_cast<const __half*>(input), indices,
      reinterpret_cast<const __half*>(values), update_count, rows, columns,
      broadcast_values, stream);
}
extern "C" void launch_row_clone_scatter_bf16(unsigned short* output,
    const unsigned short* input, const long long* indices,
    const unsigned short* values, int update_count, int rows, int columns,
    bool broadcast_values, musaStream_t stream) {
  launch_row_clone_scatter_impl(reinterpret_cast<__mt_bfloat16*>(output),
      reinterpret_cast<const __mt_bfloat16*>(input), indices,
      reinterpret_cast<const __mt_bfloat16*>(values), update_count, rows,
      columns, broadcast_values, stream);
}
extern "C" void launch_row_clone_scatter_f32(float* output, const float* input,
    const long long* indices, const float* values, int update_count, int rows,
    int columns, bool broadcast_values, musaStream_t stream) {
  launch_row_clone_scatter_impl(output, input, indices, values, update_count,
      rows, columns, broadcast_values, stream);
}

struct Offset3 { long long s0, s1, s2; };

__device__ __forceinline__ long long broadcast_offset(
    long long c0, long long c1, long long c2, int rank, Offset3 stride) {
  if (rank == 1) return c0 * stride.s0;
  if (rank == 2) return c0 * stride.s0 + c1 * stride.s1;
  return c0 * stride.s0 + c1 * stride.s1 + c2 * stride.s2;
}

template <typename T>
__global__ void scatter_fullrank(
    T* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const T* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input) {
  long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= count) return;

  long long c0 = 0, c1 = 0, c2 = 0;
  if (broadcast_rank == 1) {
    c0 = linear;
  } else if (broadcast_rank == 2) {
    c0 = linear / shape1;
    c1 = linear % shape1;
  } else {
    long long inner = (long long)shape1 * shape2;
    c0 = linear / inner;
    long long rest = linear % inner;
    c1 = rest / shape2;
    c2 = rest % shape2;
  }

  bool valid = true;
  long long target = 0;
  if (input_rank > 0) {
    long long i = 0;
    long long raw = idx0[broadcast_offset(c0, c1, c2, broadcast_rank, stride_idx0)];
    if (!normalize_index(raw, dim0, &i)) valid = false;
    else target += i * stride_input.s0;
  }
  if (input_rank > 1) {
    long long i = 0;
    long long raw = idx1[broadcast_offset(c0, c1, c2, broadcast_rank, stride_idx1)];
    if (!normalize_index(raw, dim1, &i)) valid = false;
    else target += i * stride_input.s1;
  }
  if (input_rank > 2) {
    long long i = 0;
    long long raw = idx2[broadcast_offset(c0, c1, c2, broadcast_rank, stride_idx2)];
    if (!normalize_index(raw, dim2, &i)) valid = false;
    else target += i * stride_input.s2;
  }
  if (valid) {
    long long value_offset = broadcast_offset(c0, c1, c2, broadcast_rank, stride_values);
    input[target] = values[value_offset];
  }
}

template <typename T>
__global__ void scatter_fullrank_atomic(
    T* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const T* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input) {
  long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= count) return;

  long long c0 = 0, c1 = 0, c2 = 0;
  if (broadcast_rank == 1) {
    c0 = linear;
  } else if (broadcast_rank == 2) {
    c0 = linear / shape1;
    c1 = linear % shape1;
  } else {
    long long inner = (long long)shape1 * shape2;
    c0 = linear / inner;
    long long rest = linear % inner;
    c1 = rest / shape2;
    c2 = rest % shape2;
  }

  bool valid = true;
  long long target = 0;
  if (input_rank > 0) {
    long long i = 0;
    long long raw = idx0[broadcast_offset(c0, c1, c2, broadcast_rank, stride_idx0)];
    if (!normalize_index(raw, dim0, &i)) valid = false;
    else target += i * stride_input.s0;
  }
  if (input_rank > 1) {
    long long i = 0;
    long long raw = idx1[broadcast_offset(c0, c1, c2, broadcast_rank, stride_idx1)];
    if (!normalize_index(raw, dim1, &i)) valid = false;
    else target += i * stride_input.s1;
  }
  if (input_rank > 2) {
    long long i = 0;
    long long raw = idx2[broadcast_offset(c0, c1, c2, broadcast_rank, stride_idx2)];
    if (!normalize_index(raw, dim2, &i)) valid = false;
    else target += i * stride_input.s2;
  }
  if (valid) {
    long long value_offset = broadcast_offset(c0, c1, c2, broadcast_rank, stride_values);
    atomicAdd(input + target, values[value_offset]);
  }
}

template <typename T>
static void launch_fullrank_impl(
    T* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const T* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input, musaStream_t stream) {
  scatter_fullrank<<<(count + 255) / 256, 256, 0, stream>>>(
      input, idx0, idx1, idx2, values, count, broadcast_rank,
      shape0, shape1, shape2, input_rank, dim0, dim1, dim2,
      stride_idx0, stride_idx1, stride_idx2, stride_values, stride_input);
}

extern "C" void launch_fullrank_u16(
    unsigned short* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const unsigned short* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input, musaStream_t stream) {
  launch_fullrank_impl(input, idx0, idx1, idx2, values, count,
      broadcast_rank, shape0, shape1, shape2, input_rank, dim0, dim1, dim2,
      stride_idx0, stride_idx1, stride_idx2, stride_values, stride_input, stream);
}

extern "C" void launch_fullrank_f32(
    float* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const float* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input, musaStream_t stream) {
  launch_fullrank_impl(input, idx0, idx1, idx2, values, count,
      broadcast_rank, shape0, shape1, shape2, input_rank, dim0, dim1, dim2,
      stride_idx0, stride_idx1, stride_idx2, stride_values, stride_input, stream);
}

extern "C" void launch_fullrank_atomic_u16(
    unsigned short* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const unsigned short* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input, int is_bfloat16,
    musaStream_t stream) {
  // FP16 and BF16 share a 16-bit representation but only FP16 atomicAdd is
  // implemented here. The dispatcher currently calls this for FP16 only.
  if (is_bfloat16) return;
  scatter_fullrank_atomic<<<(count + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<__half*>(input), idx0, idx1, idx2,
      reinterpret_cast<const __half*>(values), count, broadcast_rank,
      shape0, shape1, shape2, input_rank, dim0, dim1, dim2,
      stride_idx0, stride_idx1, stride_idx2, stride_values, stride_input);
}

extern "C" void launch_fullrank_atomic_f32(
    float* input, const long long* idx0, const long long* idx1,
    const long long* idx2, const float* values, long long count,
    int broadcast_rank, int shape0, int shape1, int shape2,
    int input_rank, int dim0, int dim1, int dim2,
    Offset3 stride_idx0, Offset3 stride_idx1, Offset3 stride_idx2,
    Offset3 stride_values, Offset3 stride_input, musaStream_t stream) {
  scatter_fullrank_atomic<<<(count + 255) / 256, 256, 0, stream>>>(
      input, idx0, idx1, idx2, values, count, broadcast_rank,
      shape0, shape1, shape2, input_rank, dim0, dim1, dim2,
      stride_idx0, stride_idx1, stride_idx2, stride_values, stride_input);
}

__global__ void scatter_atomic_f32(
    float* input, const long long* indices, const float* values,
    long long count, long long size) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  long long index = 0;
  if (normalize_index(indices[i], size, &index)) atomicAdd(input + index, values[i]);
}

__global__ void scatter_atomic_f16(
    __half* input, const long long* indices, const __half* values,
    long long count, long long size) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  long long index = 0;
  if (normalize_index(indices[i], size, &index)) atomicAdd(input + index, values[i]);
}

extern "C" void launch_atomic_f32(
    float* input, const long long* indices, const float* values,
    long long count, long long size, musaStream_t stream) {
  scatter_atomic_f32<<<(count + 255) / 256, 256, 0, stream>>>(
      input, indices, values, count, size);
}

extern "C" void launch_atomic_f16(
    unsigned short* input, const long long* indices,
    const unsigned short* values, long long count, long long size,
    musaStream_t stream) {
  scatter_atomic_f16<<<(count + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<__half*>(input), indices,
      reinterpret_cast<const __half*>(values), count, size);
}

__global__ void bool_mask_count(const bool* mask, int* block_counts,
                                long long size) {
  extern __shared__ int shared[];
  int* a = shared;
  int* b = shared + 256;
  int lane = threadIdx.x;
  long long linear = (long long)blockIdx.x * 256 + lane;
  a[lane] = linear < size && mask[linear] ? 1 : 0;
  __syncthreads();
  int* current = a;
  int* next = b;
  for (int offset = 1; offset < 256; offset <<= 1) {
    int left = lane >= offset ? current[lane - offset] : 0;
    next[lane] = current[lane] + left;
    __syncthreads();
    int* temporary = current;
    current = next;
    next = temporary;
  }
  if (lane == 255) block_counts[blockIdx.x] = current[255];
}

__global__ void bool_mask_block_prefix(const int* block_counts,
                                       int* block_offsets,
                                       int block_count) {
  if (threadIdx.x != 0) return;
  int running = 0;
  for (int block = 0; block < block_count; ++block) {
    block_offsets[block] = running;
    running += block_counts[block];
  }
  block_offsets[block_count] = running;
}

template <typename T>
__global__ void bool_mask_scatter(T* output, const bool* mask,
                                  const T* values, const int* block_offsets,
                                  long long size, long long value_count,
                                  bool accumulate) {
  extern __shared__ int shared[];
  int* a = shared;
  int* b = shared + 256;
  int lane = threadIdx.x;
  long long linear = (long long)blockIdx.x * 256 + lane;
  bool selected = linear < size && mask[linear];
  a[lane] = selected ? 1 : 0;
  __syncthreads();
  int* current = a;
  int* next = b;
  for (int offset = 1; offset < 256; offset <<= 1) {
    int left = lane >= offset ? current[lane - offset] : 0;
    next[lane] = current[lane] + left;
    __syncthreads();
    int* temporary = current;
    current = next;
    next = temporary;
  }
  if (selected) {
    long long value_index = (long long)block_offsets[blockIdx.x] + current[lane] - 1;
    // The C++ entry validates the device-computed True count before launch;
    // retain a bounds guard so a malformed call cannot read past Values.
    if (value_index < value_count) {
      if (accumulate) output[linear] = output[linear] + values[value_index];
      else output[linear] = values[value_index];
    }
  }
}

extern "C" void launch_bool_mask_prepare(const bool* mask,
                                          int* block_counts,
                                          int* block_offsets,
                                          long long size, int blocks,
                                          musaStream_t stream) {
  constexpr int threads = 256;
  constexpr size_t shared_bytes = 2 * threads * sizeof(int);
  bool_mask_count<<<blocks, threads, shared_bytes, stream>>>(
      mask, block_counts, size);
  bool_mask_block_prefix<<<1, 1, 0, stream>>>(
      block_counts, block_offsets, blocks);
}

template <typename T>
static void launch_bool_mask_scatter_impl(T* output, const bool* mask,
                                          const T* values,
                                          const int* block_offsets,
                                          long long size,
                                          long long value_count, int blocks,
                                          bool accumulate,
                                          musaStream_t stream) {
  constexpr int threads = 256;
  constexpr size_t shared_bytes = 2 * threads * sizeof(int);
  bool_mask_scatter<<<blocks, threads, shared_bytes, stream>>>(
      output, mask, values, block_offsets, size, value_count, accumulate);
}

extern "C" void launch_bool_mask_f16(
    unsigned short* output, const bool* mask, const unsigned short* values,
    const int* block_offsets, long long size, long long value_count,
    int blocks, bool accumulate, musaStream_t stream) {
  launch_bool_mask_scatter_impl(reinterpret_cast<__half*>(output), mask,
      reinterpret_cast<const __half*>(values), block_offsets, size,
      value_count, blocks, accumulate, stream);
}

extern "C" void launch_bool_mask_bf16(
    unsigned short* output, const bool* mask, const unsigned short* values,
    const int* block_offsets, long long size, long long value_count,
    int blocks, bool accumulate, musaStream_t stream) {
  launch_bool_mask_scatter_impl(reinterpret_cast<__mt_bfloat16*>(output), mask,
      reinterpret_cast<const __mt_bfloat16*>(values), block_offsets, size,
      value_count, blocks, accumulate, stream);
}

extern "C" void launch_bool_mask_f32(
    float* output, const bool* mask, const float* values,
    const int* block_offsets, long long size, long long value_count,
    int blocks, bool accumulate, musaStream_t stream) {
  launch_bool_mask_scatter_impl(output, mask, values, block_offsets, size,
      value_count, blocks, accumulate, stream);
}
