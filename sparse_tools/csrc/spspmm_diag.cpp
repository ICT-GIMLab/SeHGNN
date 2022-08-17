#include <Python.h>
#include <torch/script.h>

#include "cpu/spspmm_diag_cpu.h"

// #ifdef WITH_CUDA
// #include "cuda/spspmm_diag_cuda.h"
// #endif

#ifdef _WIN32
// #ifdef WITH_CUDA
// PyMODINIT_FUNC PyInit__spspmm_diag_cuda(void) { return NULL; }
// #else
PyMODINIT_FUNC PyInit__spspmm_diag_cpu(void) { return NULL; }
// #endif
#endif


torch::Tensor
spspmm_diag_sym_AAA(torch::Tensor rowptr, torch::Tensor col,
                    torch::Tensor weight, int64_t num_threads) {
  if (rowptr.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return spspmm_cuda(rowptrA, colA, optional_valueA, rowptrB, colB,
//                        optional_valueB, K, "sum");
// #else
    AT_ERROR("Not compiled with CUDA support");
// #endif
  } else {
    return spspmm_diag_sym_AAA_cpu(rowptr, col, weight, num_threads);
  }
}


torch::Tensor
spspmm_diag_sym_ABA(torch::Tensor rowptr, torch::Tensor col,
                    torch::Tensor row_weight, torch::Tensor col_weight, int64_t num_threads) {
  if (rowptr.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return spspmm_cuda(rowptrA, colA, optional_valueA, rowptrB, colB,
//                        optional_valueB, K, "sum");
// #else
    AT_ERROR("Not compiled with CUDA support");
// #endif
  } else {
    return spspmm_diag_sym_ABA_cpu(rowptr, col, row_weight, col_weight, num_threads);
  }
}


torch::Tensor
spspmm_diag_sym_AAAA(torch::Tensor rowptr, torch::Tensor col,
                     torch::Tensor weight, int64_t num_threads) {
  if (rowptr.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return spspmm_cuda(rowptrA, colA, optional_valueA, rowptrB, colB,
//                        optional_valueB, K, "sum");
// #else
    AT_ERROR("Not compiled with CUDA support");
// #endif
  } else {
    return spspmm_diag_sym_AAAA_cpu(rowptr, col, weight, num_threads);
  }
}


static auto registry =
    torch::RegisterOperators()
        .op("sparse_tools::spspmm_diag_sym_AAA", &spspmm_diag_sym_AAA)
        .op("sparse_tools::spspmm_diag_sym_ABA", &spspmm_diag_sym_ABA)
        .op("sparse_tools::spspmm_diag_sym_AAAA", &spspmm_diag_sym_AAAA);
