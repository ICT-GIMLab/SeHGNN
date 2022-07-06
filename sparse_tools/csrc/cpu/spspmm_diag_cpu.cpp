#include "spspmm_diag_cpu.h"
#include "utils.h"

// #include <iostream>
#include <omp.h>
#include <unordered_set>

#define scalar_t float


torch::Tensor
spspmm_diag_sym_AAA_cpu(torch::Tensor rowptr, torch::Tensor col,
                        torch::Tensor weight, int64_t num_threads) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(weight);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(weight.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  // TODO at::internal::lazy_init_num_threads(); // Initialise num_threads lazily at first parallel call
  omp_set_num_threads(num_threads);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto value_diag = torch::zeros({weight.numel()}, weight.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, weight.scalar_type(), "spspmm_diag_sym_AAA", [&] {
    const scalar_t* weight_data = weight.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, col -> r, e, c
#pragma omp parallel for private(r)
    for (r = 0; r < weight.numel(); r++) {
      scalar_t tmp = 0;
      for (auto e = rowptr_data[r]; e < rowptr_data[r+1]; e++) {
        tmp += weight_data[col_data[e]];
      }
      value_data[r] = weight_data[r] * tmp;
    }
  // });
  return value_diag;
}


torch::Tensor
spspmm_diag_sym_ABA_cpu(torch::Tensor rowptr, torch::Tensor col,
                        torch::Tensor row_weight, torch::Tensor col_weight, int64_t num_threads) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(row_weight);
  CHECK_CPU(col_weight);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(row_weight.dim() == 1);
  CHECK_INPUT(col_weight.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto value_diag = torch::zeros({row_weight.numel()}, row_weight.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, row_weight.scalar_type(), "spspmm_diag_sym_ABA", [&] {
    const scalar_t* row_weight_data = row_weight.data_ptr<scalar_t>();
    const scalar_t* col_weight_data = col_weight.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, col -> r, e, c
#pragma omp parallel for private(r)
    for (r = 0; r < row_weight.numel(); r++) {
      scalar_t tmp = 0;
      for (auto e = rowptr_data[r]; e < rowptr_data[r+1]; e++) {
        tmp += col_weight_data[col_data[e]];
      }
      value_data[r] = row_weight_data[r] * tmp;
    }
  // });
  return value_diag;
}


torch::Tensor
spspmm_diag_sym_AAAA_cpu(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor weight, int64_t num_threads) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(weight);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(weight.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto value_diag = torch::zeros({weight.numel()}, weight.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, weight.scalar_type(), "spspmm_diag_sym_AAAA", [&] {
    const scalar_t* weight_data = weight.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, middle, edge1, col -> r, e0, m, e1, c
#pragma omp parallel for private(r)
    for (r = 0; r < weight.numel(); r++) {
      std::unordered_set<int64_t> neighbors;
      scalar_t tmp0 = 0, tmp1 = 0;
      for (auto e0 = rowptr_data[r]; e0 < rowptr_data[r+1]; e0++) {
        neighbors.emplace(col_data[e0]);
      }
      for (auto e0 = rowptr_data[r]; e0 < rowptr_data[r+1]; e0++) {
        auto m = col_data[e0];
        tmp1 = 0;
        for (auto e1 = rowptr_data[m]; e1 < rowptr_data[m+1]; e1++) {
          auto c = col_data[e1];
          if (neighbors.find(c) != neighbors.end()) {
            tmp1 += weight_data[c];
          }
        }
        tmp0 += weight_data[m] * tmp1;
      }
      value_data[r] = weight_data[r] * tmp0;
    }
  // });
  return value_diag;
}
