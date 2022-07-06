#include "spspmm_diag_unsym_cpu.h"
#include "utils.h"

// #include <iostream>
#include <omp.h>
#include <unordered_set>

#define scalar_t float


torch::Tensor
spspmm_diag_ABA_cpu(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor weightA,
                    torch::Tensor rowptrB, torch::Tensor colB, torch::Tensor weightB, int64_t num_threads) {
  CHECK_CPU(rowptrA);
  CHECK_CPU(colA);
  CHECK_CPU(weightA);
  CHECK_CPU(rowptrB);
  CHECK_CPU(colB);
  CHECK_CPU(weightB);

  CHECK_INPUT(rowptrA.dim() == 1);
  CHECK_INPUT(colA.dim() == 1);
  CHECK_INPUT(weightA.dim() == 1);
  CHECK_INPUT(rowptrB.dim() == 1);
  CHECK_INPUT(colB.dim() == 1);
  CHECK_INPUT(weightB.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptrA_data = rowptrA.data_ptr<int64_t>();
  auto colA_data = colA.data_ptr<int64_t>();
  auto rowptrB_data = rowptrB.data_ptr<int64_t>();
  auto colB_data = colB.data_ptr<int64_t>();

  auto value_diag = torch::zeros({weightA.numel()}, weightA.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, weightA.scalar_type(), "spspmm_diag_ABA_cpu", [&] {
    const scalar_t* weightA_data = weightA.data_ptr<scalar_t>();
    const scalar_t* weightB_data = weightB.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, col -> r, e0, m, e1, c
#pragma omp parallel for private(r)
    for (r = 0; r < weightA.numel(); r++) {
      scalar_t tmp = 0;
      for (auto e0 = rowptrA_data[r]; e0 < rowptrA_data[r+1]; e0++) {
        auto m = colA_data[e0];
        for (auto e1 = rowptrB_data[m]; e1 < rowptrB_data[m+1]; e1++) {
          if (r == colB_data[e1]) {
            tmp += weightB_data[m];
            break;
          }
        }
      }
      value_data[r] = weightA_data[r] * tmp;
    }
  // });
  return value_diag;
}
