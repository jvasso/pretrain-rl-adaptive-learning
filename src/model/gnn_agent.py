from typing import List, Dict, Union, Tuple, Callable
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


import torch
from torch import nn
from torch import cat, from_numpy, sum, mean
import torch.nn.functional as F
from torch_geometric.nn.dense import Linear, HeteroDictLinear, HeteroLinear
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv, GeneralConv, TransformerConv, HGTConv, HeteroConv

from ..student_simulation.corpus_graph import CorpusGraph
from .architectures import load_arch


BASE_GNN_ARCH = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"}
]
BASE_FEEDBACK_ARCH = [
    {"layer_type": "feedback", "layer_cls": "Linear"},
    {"layer_type": "feedback", "layer_cls": "Linear"}
]
BASE_LAYERS_PARAMS = {
    "Linear"          : dict(bias=True, weight_initializer=None, bias_initializer=None),
    "HeteroDictLinear": dict(bias=True, weight_initializer=None, bias_initializer=None),
    "HGTConv"         : dict(heads=2, group="mean"),
    "TransformerConv" : dict(aggr="mean", heads=None, concat=None, beta=None, dropout=None),
    "GeneralConv"     : dict(aggr="mean", skip_linear=False, directed_msg=True, heads=2, attention=True, attention_type="additive", bias=True, l2_normalize=False)
}

DEFAULT_GNN_PARAMS = dict(kw_features_size=100, feedback_size=5, prediction_type='next_feedback', gnn_arch='arch11',
                          feedback_arch='arch2', gnn_act='elu', feedback_act='relu', hidden_dim=32, heads=1, layers_params='standard',
                          dropout_mode='last_only',dropout_conv=0.6, dropout_dense=0.2, dropout_f=0.4, aggr='mean')



class GNNAgent(nn.Module):

    DENSE_LAYERS   :Dict[str,torch.nn.Module] = {"Linear":Linear, "HeteroDictLinear":HeteroDictLinear}
    CONV_LAYERS    :Dict[str,torch.nn.Module] = {"GATConv":GATConv, "GCNConv":GCNConv, "GeneralConv":GeneralConv, "TransformerConv":TransformerConv, "HGTConv":HGTConv}
    ACTIVATION_DICT:Dict[str, Callable]       = {"relu":F.relu, "elu":F.elu, "tanh":F.tanh }
    HETERO_LAYERS  = {'HeteroDictLinear', 'HGTConv', 'HeteroConv'}
    MERGING_LAYERS = {"Mul":torch.mul, "Add":torch.add, "Sub":torch.sub}
    STR2CLS = {**DENSE_LAYERS, **CONV_LAYERS, **MERGING_LAYERS}

    NEXT_FEEDBACK = 'next_feedback'
    NEXT_RECO = 'next_reco'
    PREDICTION_TYPES = {NEXT_FEEDBACK, NEXT_RECO}
    
    def __init__(self,
                 kw_features_size:int,
                 feedback_size:int,
                 prediction_type:str,
                 gnn_arch:str,
                 feedback_arch:str,
                 gnn_act:str,
                 feedback_act:str,
                 hidden_dim:Union[int,str],
                 heads:int,
                 layers_params:str,
                 f_dropout_mode:str,
                 gnn_dropout_mode:str,
                 dropout_features:float,
                 dropout_conv:float,
                 dropout_dense:float,
                 dropout_f:float,
                 aggr:str,
                 concat:bool=None,
                 beta:bool=None,
                 edge_dropout:float=None,
                 data_sample=None,
                 device="cpu",
                 verbose:int=0):
        super().__init__()

        self.gnn_arch      = load_arch(gnn_arch, type="gnn")           if gnn_arch!="default"      else BASE_GNN_ARCH
        self.feedback_arch = load_arch(feedback_arch, type="feedback") if feedback_arch!="default" else BASE_FEEDBACK_ARCH

        self.prediction_type  = prediction_type
        self.kw_features_size = kw_features_size
        self.feebdack_size    = feedback_size
        self.f_dropout_mode   = f_dropout_mode
        self.gnn_dropout_mode = gnn_dropout_mode
        self.dropout_features = dropout_features
        self.dropout_conv     = dropout_conv
        self.dropout_dense    = dropout_dense
        self.dropout_f        = dropout_f
        self.device           = device
        self.data_sample      = data_sample
        self.verbose          = verbose
        self.layers_params    = self.preprocess_layer_params(layers_params=layers_params, params_dict={'aggr':aggr,'heads':heads,'dropout':edge_dropout,'concat':concat,'beta':beta})

        self.hidden_dim = self.kw_features_size if hidden_dim=="default" else hidden_dim

        self.gnn_act = self.ACTIVATION_DICT[gnn_act]
        self.f_act   = self.ACTIVATION_DICT[feedback_act]

        self.gnn_moduleList      = self.build_layers(arch=self.gnn_arch     , input_dim=self.kw_features_size)
        self.feedback_moduleList = self.build_layers(arch=self.feedback_arch, input_dim=self.feebdack_size)

        in_channels  = self.hidden_dim
        out_channels = self.set_final_out_channels()
        self.last    = Linear(in_channels, out_channels, bias=True)

        self.num_calls = 0

    
    def build_layers(self, arch:List[dict], input_dim) -> torch.nn.ModuleList:
        moduleList = torch.nn.ModuleList()
        is_initialized = {'kw':False, 'doc':False, 'feedback':False, 'doc_embeddings':False}
        for idx in range(len(arch)):
            layer_dict = arch[idx]
            layer_cls_name = layer_dict['layer_cls']
            layer_type     = layer_dict['layer_type']
            layer_cls      = GNNAgent.STR2CLS[layer_cls_name]
            source_nodes, target_nodes = self.get_source_target_nodes(layer_dict=layer_dict)
            accepted_nodes = source_nodes + target_nodes
            if layer_cls_name in {"Mul","Sub"}:
                layer = None
            else :
                assert all(is_initialized[target_node]==is_initialized[target_nodes[0]] for target_node in target_nodes)
                raw_in_channels = input_dim if not is_initialized[target_nodes[0]] else self.hidden_dim
                # in_channels     = { target_node:raw_in_channels for target_node in accepted_nodes } if self.is_hetero(layer_cls_name) else raw_in_channels
                in_channels     = { target_node:-1 for target_node in accepted_nodes } if self.is_hetero(layer_cls_name) else -1
                divide_factor = self.get_out_channels_factor(layer_cls_name)
                out_channels    = self.hidden_dim // divide_factor
                channels_dict   = {'in_channels':in_channels, 'out_channels':out_channels}
                if layer_cls_name == 'HGTConv':
                    assert self.data_sample is not None
                    self.layers_params[layer_cls_name]['metadata'] = self.data_sample.metadata()

                if self.is_hetero(layer_cls_name) or self.is_dense_layer(layer_cls_name):
                    layer = layer_cls(**{**channels_dict.copy(), **self.layers_params[layer_cls_name]})
                else:
                    accepted_edge_index_list = self.get_accepted_edge_index(layer_type=layer_type)
                    convs = { accepted_edge_index:layer_cls(**{**channels_dict.copy(), **self.layers_params[layer_cls_name]}) for accepted_edge_index in accepted_edge_index_list }
                    layer = HeteroConv(convs=convs, aggr='sum')
                is_initialized.update({target_node:True for target_node in target_nodes})
            
            moduleList.append(layer)
        return moduleList
    

    def get_out_channels_factor(self, layer_cls_name):
        layer_param = self.layers_params[layer_cls_name]
        if 'heads' in layer_param:
            if 'concat' in layer_param:
                if layer_param['concat']:
                    return layer_param['heads']
        return 1
    
    
    def forward(self, x_dict:Dict[str, torch.Tensor], edge_index_dict:Dict[Tuple[str,str,str], torch.Tensor], corpus_graph_list:List[CorpusGraph]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x_dict: keys: {"kw", "doc", "feedback"} ; values: torch.Tensor
        :param edge_index_dict: keys: {("doc","to","kw"), ("kw","to_all","doc")} ; values: torch.Tensor

        :return: _description_
        """
        self.num_calls += 1

        # convert data
        x_dict["feedback"] = x_dict["feedback"].to(torch.float32)
        edge_index_dict["doc","to","kw"]     = edge_index_dict["doc","to","kw"].to(torch.int64)
        edge_index_dict["kw","to2","doc"]    = edge_index_dict["kw","to2","doc"].to(torch.int64)
        edge_index_dict["kw","to_all","doc"] = edge_index_dict["kw","to_all","doc"].to(torch.int64)

        if self.dropout_features > 0:
            dropout_params_features = {"p":self.dropout_features, "training":self.training}
            x_dict['kw']  = F.dropout(x_dict['kw'], **dropout_params_features)
            x_dict['doc'] = F.dropout(x_dict['doc'], **dropout_params_features)
        
        # forward
        x_dict = self._forward_feedback_net(x_dict=x_dict, corpus_graph_list=corpus_graph_list)
        x_dict = self._forward_gnn(x_dict=x_dict, edge_index_dict=edge_index_dict, corpus_graph_list=corpus_graph_list)
        
        out = self.last(x_dict["doc"])
        return out, x_dict["kw"]
    

    def _forward_feedback_net(self, x_dict:Dict[str, torch.Tensor], corpus_graph_list:List[CorpusGraph]=None) -> Dict[str, torch.Tensor]:
        dropout_params_feedback = {"p":self.dropout_f, "training":self.training}
        for idx in range(len(self.feedback_arch)):
            layer = self.feedback_moduleList[idx]
            x_dict['feedback'] = layer(x_dict['feedback'])
            if idx<len(self.feedback_arch)-1:
                if self.f_act is not None:
                    x_dict['feedback'] = self.f_act(x_dict['feedback'])
                if self.f_dropout_mode == 'all_except_last':
                    x_dict['feedback'] = F.dropout(x_dict['feedback'], **dropout_params_feedback)
        return x_dict


    def _forward_gnn(self, x_dict:Dict[str,torch.Tensor], edge_index_dict:Dict[Tuple[str,str,str],torch.Tensor], corpus_graph_list:List[CorpusGraph]=None) -> Dict[str, torch.Tensor]:
        have_merged_feedback = False
        doc_embeddings       = None

        for idx in range(len(self.gnn_arch)):
            layer_dict     = self.gnn_arch[idx]
            layer_cls_name = layer_dict['layer_cls']
            layer          = self.gnn_moduleList[idx]
            source_nodes, target_nodes = self.get_source_target_nodes(layer_dict=layer_dict)
            is_last = idx == len(self.gnn_arch)-1

            if self.is_dense_layer(layer_cls_name):
                x_dict, doc_embeddings = self._forward_dense_layer(x_dict=x_dict, doc_embeddings=doc_embeddings, layer=layer, layer_dict=layer_dict, have_merged_feedback=have_merged_feedback,
                                                                   target_nodes=target_nodes, is_last=is_last)
            elif self.is_conv_layer(layer_cls_name):
                x_dict, doc_embeddings = self._forward_conv_layer(x_dict=x_dict, doc_embeddings=doc_embeddings, layer=layer, layer_dict=layer_dict, have_merged_feedback=have_merged_feedback,
                                                                  target_nodes=target_nodes, source_nodes=source_nodes, edge_index_dict=edge_index_dict, is_last=is_last)
            elif self.is_merging_layer(layer_cls_name):
                x_dict, doc_embeddings, have_merged_feedback = self._forward_merge_layer(x_dict=x_dict, doc_embeddings=doc_embeddings, layer_dict=layer_dict, have_merged_feedback=have_merged_feedback, is_last=is_last)
            else:
                raise ValueError(f'Layer class "{layer_cls_name}" not supported.')
        return x_dict
    

    def _forward_dense_layer(self, x_dict:Dict[str, torch.Tensor], doc_embeddings:torch.Tensor, layer:torch.nn.Module, layer_dict:dict, have_merged_feedback:bool, target_nodes:List[str], is_last:bool):
        dropout_params_dense = {"p":self.dropout_dense, "training":self.training}
        if len(target_nodes) > 0:
            x_dict_restrict = {node_type:val for node_type,val in x_dict.items() if node_type in target_nodes}
        
        if self.is_hetero(layer_dict['layer_cls']):
            x_dict_restrict = layer(x_dict=x_dict_restrict)
            x_dict.update(x_dict_restrict)
            x_dict.update({key:self.gnn_act(x_dict[key]) for key in target_nodes})
            if not have_merged_feedback: doc_embeddings = x_dict["doc"]
            if (self.gnn_dropout_mode in {'all', 'all_except_conv'}) or ((self.gnn_dropout_mode=='last_only') and is_last):
                x_dict.update({key:F.dropout(x_dict[key], **dropout_params_dense) for key in target_nodes})
        else:
            for target_node in target_nodes:
                x_dict[target_node] = layer(x_dict[target_node])
                x_dict[target_node] = self.gnn_act(x_dict[target_node])
                if not have_merged_feedback and target_node=='doc':
                    doc_embeddings = x_dict["doc"]
                if (self.gnn_dropout_mode in {'all', 'all_except_conv'}) or ((self.gnn_dropout_mode=='last_only') and is_last):
                    x_dict[target_node] = F.dropout(x_dict[target_node], **dropout_params_dense)
        return x_dict, doc_embeddings
    

    def _forward_conv_layer(self, x_dict:Dict[str,torch.Tensor], doc_embeddings:torch.Tensor, layer:MessagePassing, layer_dict:dict, have_merged_feedback:bool, target_nodes:List[str], source_nodes:List[str], edge_index_dict:Dict[Tuple[str,str,str],torch.Tensor], is_last:bool):
        layer_cls = layer_dict['layer_cls']
        layer_type = layer_dict['layer_type']
        dropout_params_conv = {"p":self.dropout_conv, "training":self.training}
        
        if 'doc_mode' in layer_dict.keys():
            if layer_dict['doc_mode']=='doc_embeddings':
                x_dict['doc'] = doc_embeddings # check that these are different first
            else:
                raise ValueError(f'Doc mode "{layer_dict["doc_mode"]}" not supported.')

        if self.is_hetero(layer_cls) or isinstance(layer,HeteroConv):
            accepted_nodes      = source_nodes+target_nodes
            accepted_edge_index = self.get_accepted_edge_index(layer_type=layer_type)
            x_dict_restrict          = {node_type:val for node_type, val in x_dict.items() if node_type in accepted_nodes}
            edge_index_dict_restrict = {key:edge_index_dict[key] for key in accepted_edge_index}
            x_dict_restrict = layer(x_dict=x_dict_restrict, edge_index_dict=edge_index_dict_restrict)
            x_dict.update({key:x_dict_restrict[key]      for key in target_nodes})
            x_dict.update({key:self.gnn_act(x_dict[key]) for key in target_nodes})
            if not have_merged_feedback: doc_embeddings = x_dict['doc']
            if (self.gnn_dropout_mode=='all') or ((self.gnn_dropout_mode=='last_only') and is_last):
                x_dict.update({key:F.dropout(x_dict[key], **dropout_params_conv) for key in target_nodes})
        else:
            raise NotImplementedError()
            # assert len(target_nodes)==1
            # target_node = target_nodes[0]
            # flow = self.get_flow(layer_type)
            # accepted_edge_index = self.get_accepted_edge_index(layer_type=layer_type)
            # assert len(accepted_edge_index)==0
            # edge_index = accepted_edge_index[0]
            # x_dict[target_node] = layer(x=x_dict[target_node], edge_index=edge_index, flow=flow)
            # x_dict[target_node] = self.gnn_act(x_dict[target_node])
            # if not have_merged_feedback: doc_embeddings = x_dict['doc']
            # if self.dropout_mode=='all': x_dict[target_node] = F.dropout(x_dict[target_node], **dropout_params_conv)
        return x_dict, doc_embeddings
    

    def _forward_merge_layer(self, x_dict:Dict[str,torch.Tensor], doc_embeddings:torch.Tensor, layer_dict:str, have_merged_feedback:bool, is_last:bool):
        layer_cls = layer_dict['layer_cls']
        merge_operator = GNNAgent.MERGING_LAYERS[layer_cls]
        if layer_cls=="Mul":
            x_dict["doc"] = merge_operator(x_dict["doc"], x_dict["feedback"])
            have_merged_feedback = True
        elif layer_cls=="Sub":
            x_dict["doc"] = merge_operator(x_dict["doc"], doc_embeddings) 
        else:
            raise ValueError(f'Layer class {layer_cls} not supported.')
        return x_dict, doc_embeddings, have_merged_feedback
    

    def get_source_target_nodes(self, layer_dict:Dict[str,str]) -> Tuple[List[str],List[str]]:
        layer_type:str = layer_dict['layer_type']
        if layer_type in {'merge_f', 'sub_docs'}:
            source = ['doc']
            target = ['doc']
        elif '2' in layer_type:
            if layer_type == "kw2all_doc":
                source = ['kw']
                target = ["doc"]
            else:
                source, target = layer_type.split("2")
                source, target = [source], [target]
        elif "_" in layer_type:
            source = []
            target = layer_type.split("_")
        else:
            assert layer_type in {"kw", "doc", "feedback"}
            source = []
            target = [layer_type]
        # if 'target_nodes' in layer_dict:
        #     target = layer_dict['target_nodes'] if isinstance(layer_dict['target_nodes'],list) else [layer_dict['target_nodes']]
        return source, target
    

    def get_accepted_edge_index(self, layer_type:str) -> List[Tuple[str]]:
        if layer_type=="doc2kw":
            keys = [("doc", "to", "kw")]
        elif layer_type=="kw2doc":
            keys = [("kw", "to2", "doc")]
        elif layer_type=="kw2all_doc":
            keys = [("kw", "to_all", "doc")]
        elif layer_type in {"kw_doc", "doc_kw"}:
            keys = [("doc", "to", "kw"), ("kw", "to2", "doc")]
        else:
            raise Exception(f'Layer type "{layer_type}" not supported.')
        return keys
    
    
    def get_flow(self, layer_type:str):
        if layer_type=="doc2kw":
            return "source_to_target"
        elif layer_type=="kw2doc":
            return "target_to_source"
        elif layer_type=="kw2all_doc":
            return "source_to_target"
        else:
            raise Exception(f'Layer type "{layer_type}" not supported.')
    

    def set_final_out_channels(self) -> int:
        if self.prediction_type == "next_feedback":
            return 3
        elif self.prediction_type == "next_reco":
            return 1
        else:
            raise Exception(f'Prediction type "{self.prediction_type}" not supported.')
    

    def is_conv_layer(self, layer_cls_name):
        return layer_cls_name in GNNAgent.CONV_LAYERS.keys()
    
    def is_dense_layer(self, layer_cls_name):
        return layer_cls_name in GNNAgent.DENSE_LAYERS.keys()
    
    def is_merging_layer(self, layer_cls_name):
        return layer_cls_name in GNNAgent.MERGING_LAYERS.keys()
    
    def get_middle_dim(self):
        return self.hidden_dim

    def is_hetero(self, layer_cls_name:str):
        return layer_cls_name in GNNAgent.HETERO_LAYERS
    
    def preprocess_layer_params(self, layers_params:str, params_dict:dict):
        if layers_params=='standard':
            layers_params = BASE_LAYERS_PARAMS
        else:
            raise ValueError(f'Layer params {layers_params} not implemented.')
        if "aggr" in params_dict:
            params_dict["group"] = params_dict["aggr"]
        for key,value in params_dict.items():
            for layer_name, layer_params in layers_params.items():
                if key in layer_params:
                    layers_params[layer_name][key] = value
        return layers_params
    
    def compute_model_memory_usage(self):
        total_memory = 0
        for param in self.parameters():
            param_memory = param.numel() * param.element_size()
            total_memory += param_memory
        # Convert to more readable units, like megabytes (MB)
        total_memory_MB = total_memory / (1024 ** 2)
        return total_memory_MB
    

    def set_dropout_to_zero(self):
        self.dropout_conv     = 0
        self.dropout_dense    = 0
        self.dropout_f        = 0
        self.dropout_features = 0
        self.edge_dropout     = 0
    

    def check_requires_grad(self, verbose=True):
        requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad:
                requires_grad = True
            if verbose: print(f"{name} requires_grad: {param.requires_grad}")
        return requires_grad


    def display_embeddings(self, x_dict:Dict[str, torch.Tensor], edge_index_dict:Dict[Tuple[str,str,str], torch.Tensor], corpus_graph:CorpusGraph):
        _, raw_hidden_kw, raw_hidden_doc = self(x_dict, edge_index_dict)
        tsne = TSNE(n_components=2)
        edges = edge_index_dict["doc", "to", "kw"]
        embeddings  = tsne.fit_transform(torch.cat([raw_hidden_kw, raw_hidden_doc]).detach().cpu().numpy())
        kw_last_idx = int(raw_hidden_kw.shape[0])
        doc_last_idx = kw_last_idx + int(raw_hidden_doc.shape[0])
        kw_embeddings  = embeddings[:kw_last_idx]
        doc_embeddings = embeddings[kw_last_idx:doc_last_idx]
        plt.figure(figsize=(10, 8))
        plt.scatter(kw_embeddings [:, 0], kw_embeddings [:, 1], c='blue', label='kw')
        plt.scatter(doc_embeddings[:, 0], doc_embeddings[:, 1], c='blue', label='docs')
        # Draw the edges
        for edge in edges.t():
            node1, node2 = edge
            x1, y1 = embeddings[node1]
            x2, y2 = embeddings[node2]
            plt.plot([x1, x2], [y1, y2], c='black', alpha=0.5)
        node_labels = [kw.name for kw in corpus_graph.kw_list] + [doc.name for doc in corpus_graph.doc_list]
        # Add labels to the nodes
        for i, label in enumerate(node_labels):
            x, y = embeddings[i]
            plt.text(x, y, label, color='black', fontsize=8)
        plt.title('2D Visualization of Node Embeddings with Edges')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()


if __name__=="__main__":

    supervised_learning_exp = True
    rl_exp = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if supervised_learning_exp:
        from ..student_simulation.corpus_graph_dataset import CorpusGraphDataset
        from torch_geometric.loader.dataloader import DataLoader
        from . import sl_train_eval_old
        
        exp_cfg = dict(dataset_name="first_dataset")
        gnn_cfg = dict(
            prediction_type="next_feedback",
            gnn_arch = "default",
            feedback_arch = "default",
            gnn_activation = "elu",
            hidden_dim=32
        )
        training_cfg = dict(
            batch_size_train=32,
            batch_size_test=32,
            lr=0.01,
            weight_decay=5e-4,
            num_epochs=2
        )
        config = dict(**exp_cfg, **gnn_cfg, **training_cfg)        

        train_graphs, test_graphs = CorpusGraphDataset.generate_train_eval_corpus_graphs(dataset_name=config['dataset_name'], split_mode="tiny")
        
        train_dataset = CorpusGraphDataset(corpus_list=train_graphs, dataset_name=config['dataset_name'], is_eval=False)
        test_dataset  = CorpusGraphDataset(corpus_list=test_graphs,  dataset_name=config['dataset_name'], is_eval=True)
        train_loader  = DataLoader(train_dataset, batch_size=config['batch_size_train'], shuffle=True)
        test_loader   = DataLoader(test_dataset,  batch_size=config['batch_size_test'], shuffle=False)
        data_sample = train_dataset[0]
        model = GNNAgent(prediction_type=config['prediction_type'],
                         gnn_arch=config['gnn_arch'],
                         feedback_arch=config['feedback_arch'],
                         hidden_dim=config['hidden_dim'],
                         data_sample=data_sample)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        # weight = None
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        
        for epoch in range(config['num_epochs']):
            sl_train_eval_old.train(model, train_loader, criterion, optimizer, device=device)
            train_acc = test(model=model, loader=train_loader, device=device)
            test_acc  = test(model=model, loader=test_loader, device=device)
            print(f'Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            wandb.log({"train/acc": train_acc, "test/acc":test_acc, "train/loss": train_loss, "test/loss": test_loss})