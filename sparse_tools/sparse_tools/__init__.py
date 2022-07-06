import importlib
import os.path as osp

import torch

# suffix = 'cuda' if torch.cuda.is_available() else 'cpu'
suffix = 'cpu'

for library in ['_spspmm_diag', '_spspmm_diag_unsym']:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        f'{library}_{suffix}', [osp.dirname(__file__)]).origin)

from .spspmm_diag import spspmm_diag_sym_AAA, spspmm_diag_sym_ABA, spspmm_diag_sym_AAAA
from .spspmm_diag_unsym import spspmm_diag_ABA

__all__ = [
    'spspmm_diag_sym_AAA',
    'spspmm_diag_sym_ABA',
    'spspmm_diag_sym_AAAA',
    'spspmm_diag_ABA',
]
