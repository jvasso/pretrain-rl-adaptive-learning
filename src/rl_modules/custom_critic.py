from typing import Any, Dict, Optional, Sequence, Tuple, Union
from typing import Dict, List
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from torch import nn

from ..model import GNNAgent, MLPAgent
from ..student_simulation.corpus_graph_dataset import CorpusGraphDataset

from .custom_actor import CustomActor

from torch_geometric.nn.dense import Linear
from torch_geometric.data import HeteroData
# from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch as GeometricBatch


class CustomCritic(nn.Module):
    
    def __init__(
        self,
        model:Union[GNNAgent, MLPAgent],
        device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device

        if isinstance(self.model, MLPAgent):
            in_features = self.model.output_dim
            self.last = nn.Linear(in_features=in_features, out_features=1, bias=True)
        else:
            raise ValueError(f'Model type "{type(self.model)}" not supported.')
        

    def forward(self, observations:np.ndarray, state: Any = None, info: Dict[str, Any] = {}): 
        data_batch = CustomActor.preprocess_batch(batch=observations)
        data_batch = data_batch.to(device=self.device)
        
        if isinstance(self.model, MLPAgent):
            x_feedback = data_batch.x_dict['feedback']
            # x_feedback = CustomActor.reshape_feedback_tensor(x_feedback=x_feedback)
            out = self.model(x_feedback)
            out = self.last(out)
        
        return out