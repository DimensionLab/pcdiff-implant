#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

template <typename scalar_t>
__global__ void grouping_kernel(int b, int c, int n, int m, int u,
                                const scalar_t *__restrict__ features,
                                const int *__restrict__ indices,
                                scalar_t *__restrict__ out) {
  int batch_index = blockIdx.x;
  features += batch_index * n * c;
  indices += batch_index * m * u;
  out += batch_index * m * u * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * m; i += stride) {
    const int l = i / m;
    const int j = i % m;
    for (int k = 0; k < u; ++k) {
      const int idx = indices[j * u + k];
      if (idx >= 0 && idx < n) {
        out[(l * m + j) * u + k] = features[l * n + idx];
      }
    }
  }
}

template <typename scalar_t>
__global__ void grouping_grad_kernel(int b, int c, int n, int m, int u,
                                     const scalar_t *__restrict__ grad_y,
                                     const int *__restrict__ indices,
                                     scalar_t *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  grad_y += batch_index * m * u * c;
  indices += batch_index * m * u;
  grad_x += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * m; i += stride) {
    const int l = i / m;
    const int j = i % m;
    auto grad_channel = grad_x + l * n;
    for (int k = 0; k < u; ++k) {
      const int idx = indices[j * u + k];
      if (idx >= 0 && idx < n) {
        const auto grad_val = grad_y[(l * m + j) * u + k];
        at::native::fastAtomicAdd(
            grad_channel, idx, n, grad_val,
            /*fast_atomics=*/true);
      }
    }
  }
}

template <typename scalar_t>
void grouping_forward_dispatch(int b, int c, int n, int m, int u,
                               const scalar_t *features, const int *indices,
                               scalar_t *out) {
  grouping_kernel<scalar_t><<<b, optimal_block_config(m, c), 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, m, u, features, indices, out);
  CUDA_CHECK_ERRORS();
}

template <typename scalar_t>
void grouping_backward_dispatch(int b, int c, int n, int m, int u,
                                const scalar_t *grad_y, const int *indices,
                                scalar_t *grad_x) {
  grouping_grad_kernel<scalar_t><<<b, optimal_block_config(m, c), 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, m, u, grad_y, indices, grad_x);
  CUDA_CHECK_ERRORS();
}

void grouping_forward_launcher(int b, int c, int n, int m, int u,
                               at::ScalarType dtype, const void *features,
                               const int *indices, void *out) {
  switch (dtype) {
    case at::ScalarType::Float:
      grouping_forward_dispatch<float>(b, c, n, m, u,
                                       static_cast<const float *>(features),
                                       indices, static_cast<float *>(out));
      break;
    case at::ScalarType::Half:
      grouping_forward_dispatch<at::Half>(
          b, c, n, m, u, static_cast<const at::Half *>(features), indices,
          static_cast<at::Half *>(out));
      break;
    case at::ScalarType::BFloat16:
      grouping_forward_dispatch<at::BFloat16>(
          b, c, n, m, u, static_cast<const at::BFloat16 *>(features), indices,
          static_cast<at::BFloat16 *>(out));
      break;
    default:
      AT_ERROR("grouping_forward_launcher: unsupported dtype");
  }
}

void grouping_backward_launcher(int b, int c, int n, int m, int u,
                                at::ScalarType dtype, const void *grad_y,
                                const int *indices, void *grad_x) {
  switch (dtype) {
    case at::ScalarType::Float:
      grouping_backward_dispatch<float>(b, c, n, m, u,
                                        static_cast<const float *>(grad_y),
                                        indices, static_cast<float *>(grad_x));
      break;
    case at::ScalarType::Half:
      grouping_backward_dispatch<at::Half>(
          b, c, n, m, u, static_cast<const at::Half *>(grad_y), indices,
          static_cast<at::Half *>(grad_x));
      break;
    case at::ScalarType::BFloat16:
      grouping_backward_dispatch<at::BFloat16>(
          b, c, n, m, u, static_cast<const at::BFloat16 *>(grad_y), indices,
          static_cast<at::BFloat16 *>(grad_x));
      break;
    default:
      AT_ERROR("grouping_backward_launcher: unsupported dtype");
  }
}
