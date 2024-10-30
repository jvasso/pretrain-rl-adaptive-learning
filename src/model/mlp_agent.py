
from typing import List, Dict, Union, Tuple, Callable, Sequence


from tianshou.utils.net.common import Net, MLP

import torch
from torch import nn


class MLPAgent(nn.Module):
    
    def __init__(self,
                 input_dim:Sequence[int],
                 hidden_sizes:Union[str, int, Sequence[int]],
                 device="cpu",
                 verbose:int=0):
        super().__init__()

        self.input_dim    = input_dim
        self.hidden_sizes = hidden_sizes[:-1]
        self.output_dim   = hidden_sizes[-1]
        self.device       = device

        self.net = MLP(input_dim=self.input_dim,
                       output_dim=self.output_dim,
                       hidden_sizes=self.hidden_sizes,
                       device=device)
    

    def forward(self, X_feedback:torch.Tensor):
        out = self.net(X_feedback)
        return out
