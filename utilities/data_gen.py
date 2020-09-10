import torch
import numpy as np
from torch.autograd import Variable

from .batch import Batch

def data_gen(V, batch, nbatches, device):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data = data.type(torch.LongTensor)
        data[:, 0] = 1

        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        
        yield Batch(src, tgt, 0, device)
