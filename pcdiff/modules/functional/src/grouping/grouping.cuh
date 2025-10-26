#ifndef _GROUPING_CUH
#define _GROUPING_CUH

#include <torch/extension.h>

void grouping_forward_launcher(int b, int c, int n, int m, int u,
                               at::ScalarType dtype, const void *features,
                               const int *indices, void *out);
void grouping_backward_launcher(int b, int c, int n, int m, int u,
                                at::ScalarType dtype, const void *grad_y,
                                const int *indices, void *grad_x);

#endif
