from __future__ import annotations

import math
import numpy as np
from torch.utils.data import Sampler
from replacement_a8_strict_cross_fulltarget_confirmation_v1.common_v1 import DATASET_CODE, SEED

class EpochShuffleFullCoverageBatchSamplerV1(Sampler):
    def __init__(self,n_samples,batch_size,dataset,target,base_seed=SEED):
        self.n=int(n_samples); self.batch_size=int(batch_size); self.dataset=dataset; self.target=int(target); self.base_seed=int(base_seed); self.epoch=1
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def permutation_seed(self): return self.base_seed+DATASET_CODE[self.dataset]*100000+20000+self.target*100+self.epoch
    def indices(self): return np.random.default_rng(self.permutation_seed()).permutation(self.n).astype(np.int64).tolist()
    def __iter__(self):
        idx=self.indices(); return iter([idx[i:i+self.batch_size] for i in range(0,self.n,self.batch_size)])
    def __len__(self): return int(math.ceil(self.n/self.batch_size))
