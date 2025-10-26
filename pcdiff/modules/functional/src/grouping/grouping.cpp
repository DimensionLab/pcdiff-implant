#include "grouping.hpp"
#include "grouping.cuh"

#include "../utils.hpp"

at::Tensor grouping_forward(at::Tensor features, at::Tensor indices) {
  CHECK_CUDA(features);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(indices);
  auto dtype = features.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Float || dtype == at::ScalarType::Half ||
          dtype == at::ScalarType::BFloat16,
      "features must be a float/half/bfloat16 tensor");
  CHECK_IS_INT(indices);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int m = indices.size(1);
  int u = indices.size(2);
  at::Tensor output = torch::zeros({b, c, m, u}, features.options());
  grouping_forward_launcher(b, c, n, m, u, dtype, features.data_ptr(),
                            indices.data_ptr<int>(), output.data_ptr());
  return output;
}

at::Tensor grouping_backward(at::Tensor grad_y, at::Tensor indices,
                             const int n) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  auto dtype = grad_y.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Float || dtype == at::ScalarType::Half ||
          dtype == at::ScalarType::BFloat16,
      "grad_y must be a float/half/bfloat16 tensor");
  CHECK_IS_INT(indices);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int m = indices.size(1);
  int u = indices.size(2);
  at::Tensor grad_x = torch::zeros({b, c, n}, grad_y.options());
  grouping_backward_launcher(b, c, n, m, u, dtype, grad_y.data_ptr(),
                             indices.data_ptr<int>(), grad_x.data_ptr());
  return grad_x;
}
