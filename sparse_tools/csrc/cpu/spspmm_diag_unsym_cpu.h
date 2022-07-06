#pragma once

#include <torch/extension.h>


torch::Tensor
spspmm_diag_ABA_cpu(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor weightA,
                    torch::Tensor rowptrB, torch::Tensor colB, torch::Tensor weightB, int64_t num_threads);
