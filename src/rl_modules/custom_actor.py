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

from torch_geometric.nn.dense import Linear
from torch_geometric.data import HeteroData
# from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch as GeometricBatch


class CustomActor(nn.Module):
    """Simple actor network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    """

    def __init__(
        self,
        model:Union[GNNAgent, MLPAgent],
        keep_sl_head:bool,
        freeze_layers:str,
        device = torch.device("cpu"),
        num_actions = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.keep_sl_head = keep_sl_head
        self.freeze_layers = freeze_layers
        self.device = device

        if isinstance(self.model, MLPAgent):
            assert num_actions is not None
            in_features = self.model.output_dim
            self.last = nn.Linear(in_features=in_features, out_features=num_actions, bias=True)
        
        elif isinstance(self.model, GNNAgent):
            if self.freeze_layers != 'none':
                for param in self.model.parameters():
                    param.requires_grad = False
                if self.freeze_layers=='all':
                    pass
                elif self.freeze_layers=='all_except_last_gnn_module':
                    last_module = self.model.gnn_moduleList[-1]
                    for param in last_module.parameters():
                        param.requires_grad = True
                else:
                    raise ValueError(f'Value "{self.freeze_layers}" for freeze_layers not supported.')
            
            if self.model.last.out_channels!=1:
                in_channels  = self.model.last.in_channels
                is_bias      = self.model.last.bias is not None
                out_channels = self.model.last.out_channels
                if out_channels==3:
                    if self.keep_sl_head:
                        self.model.last = nn.Sequential(self.model.last, Linear(out_channels, 1, bias=is_bias))
                    else: # replace feedback prediction with score prediction
                        self.model.last = Linear(in_channels, 1, bias=is_bias)
                        for param in self.model.last.parameters():
                            param.requires_grad = True
                elif out_channels==1:
                    pass
                else:
                    raise Exception(f'Expected out_channels to be 1 or 3 (found "{out_channels}").')
        else:
            raise ValueError(f'Model type "{type(self.model)}" not supported.')
            
    
    
    def forward(self, observations:np.ndarray, state: Any = None, info: Dict[str, Any] = {}): 
        data_batch = CustomActor.preprocess_batch(batch=observations)
        data_batch = data_batch.to(device=self.device)
        # print(f'receive obs of shape {data_batch.x_dict["feedback"].shape} (batch size: {data_batch.batch_size})')
        
        if isinstance(self.model, MLPAgent):
            x_feedback = data_batch.x_dict['feedback']
            # x_feedback = CustomActor.reshape_feedback_tensor(x_feedback)
            out = self.model(x_feedback)
            out = self.last(out)
            logits = F.softmax(out, dim=-1)
        
        elif isinstance(self.model, GNNAgent):
            out, _ = self.model(data_batch.x_dict, data_batch.edge_index_dict, corpus_graph_list=data_batch["corpus_graph"])
            
            num_docs = [corpus_graph.num_docs for corpus_graph in data_batch['corpus_graph']]
            num_docs_max = max(num_docs)
            ptr = data_batch['doc'].ptr

            out = out.squeeze()
            softmax_segments = [F.softmax(torch.cat([out[start:end], torch.full((num_docs_max-(end-start),),-float('inf')).to(self.device)]), dim=0) for start, end in zip(ptr[:-1], ptr[1:])]
            logits = torch.stack(softmax_segments, dim=0)
        
        return logits, state

    
    @staticmethod
    def preprocess_batch(batch:Union[np.ndarray, list]) -> HeteroData:
        if isinstance(batch, np.ndarray):
            batch = batch.tolist()
        else:
            assert isinstance(batch, list)
        
        if isinstance(batch[0], SimpleNamespace):
            # batch = self.convert_to_hetero_data(batch)
            preprocessed_batch = [CorpusGraphDataset.build_hetero_data(**vars(data)) for data in batch]
        else:
            raise NotImplementedError()
        
        return GeometricBatch.from_data_list(
            preprocessed_batch,
            follow_batch=None,
            exclude_keys=None
        )
    
    def convert_to_hetero_data(self, batch:List[SimpleNamespace]):
        new_batch_list = []
        for data in batch:
            data = CorpusGraphDataset.build_hetero_data(**vars(data))
            new_batch_list.append(data)
        return new_batch_list

