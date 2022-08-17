#include <Python.h>
#include <torch/script.h>

#include "cpu/spspmm_diag_unsym_cpu.h"

// #ifdef WITH_CUDA
// #include "cuda/spspmm_diag_cuda.h"
// #endif

#ifdef _WIN32
// #ifdef WITH_CUDA
// PyMODINIT_FUNC PyInit__spspmm_diag_unsym_cuda(void) { return NULL; }
// #else
PyMODINIT_FUNC PyInit__spspmm_diag_unsym_cpu(void) { return NULL; }
// #endif
#endif


torch::Tensor
spspmm_diag_ABA(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor weightA,
                torch::Tensor rowptrB, torch::Tensor colB, torch::Tensor weightB, int64_t num_threads) {
  if (rowptrA.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return spspmm_cuda(rowptrA, colA, optional_valueA, rowptrB, colB,
//                        optional_valueB, K, "sum");
// #else
    AT_ERROR("Not compiled with CUDA support");
// #endif
  } else {
    return spspmm_diag_ABA_cpu(
      rowptrA, colA, weightA, rowptrB, colB, weightB, num_threads);
  }
}


static auto registry =
    torch::RegisterOperators()
        .op("sparse_tools::spspmm_diag_ABA", &spspmm_diag_ABA);
