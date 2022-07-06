import torch
from torch_sparse import SparseTensor


# not symmetric
def spspmm_diag_ABA(AB: SparseTensor, BA: SparseTensor, num_threads: int = 4) -> torch.Tensor:
    assert AB.sparse_size(1) == BA.sparse_size(0) # 可以进行乘法
    assert AB.sparse_size(0) == BA.sparse_size(1) # 最后输出的一定是 N * N 矩阵
    rowptrA, colA, _ = AB.csr()
    rowptrB, colB, _ = BA.csr()
    weightA = 1. / AB.storage.rowcount()
    weightA[torch.isnan(weightA) | torch.isinf(weightA)] = 0.
    weightB = 1. / BA.storage.rowcount()
    weightB[torch.isnan(weightB) | torch.isinf(weightB)] = 0.

    return torch.ops.sparse_tools.spspmm_diag_ABA(
        rowptrA, colA, weightA, rowptrB, colB, weightB, num_threads)
