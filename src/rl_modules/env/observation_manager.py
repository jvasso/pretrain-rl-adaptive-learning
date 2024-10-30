import time
from typing import TYPE_CHECKING, List, Sequence
from types import SimpleNamespace
import pprint

from gymnasium import spaces

import torch
from torch_geometric.data import HeteroData

from ...student_simulation.types import Interaction, Feedback
from ...student_simulation import CorpusGraph
from ...student_simulation import CorpusGraphDataset

if TYPE_CHECKING:
    from .adaptive_learning_env import AdaptiveLearningEnv


class Observation:

    def __init__(self, 
                 env:"AdaptiveLearningEnv",
                 kw_features: torch.Tensor,
                 doc_features: torch.Tensor,
                 interact_features: torch.Tensor,
                 edge_indices_doc2kw: torch.Tensor,
                 edge_indices_kw2doc: torch.Tensor):
        self.env = env

        self.kw_features         = kw_features
        self.doc_features        = doc_features
        self.interact_features   = interact_features
        self.edge_indices_doc2kw = edge_indices_doc2kw
        self.edge_indices_kw2doc = edge_indices_kw2doc
        
        self.num_docs = self.env.num_docs
        self.num_kw   = self.env.num_kw
        self.num_edge_indices = self.edge_indices_kw2doc.shape[1]
        
        self.num_nodes = self.num_kw + self.num_docs
        self.node_features = torch.cat([self.kw_features, self.doc_features])
    

    def to_vector(self):
        return None

    
    @staticmethod
    def print_list(observations_list:List["Observation"]):
        for obs in observations_list:
            pprint.pprint(obs.interact_features)



class ObservationManager:

    def __init__(self, env:"AdaptiveLearningEnv", type='default'):
        self.env = env
        self.type = type
        self._set_observation_space()
    

    def _set_observation_space(self):
        
        # node_features_size,     node_features_range      = self.env.corpus_graph.get_features_size_and_range()
        # interact_features_size, interact_features_range  = Interaction.get_vector_size_and_range(time_mode=self.env.time_mode)
        
        # node_features_space = spaces.Box(low=node_features_range[0], high=node_features_range[1], shape=(node_features_size,), dtype=float)
        # interaction_feature_space = spaces.Box(low=interact_features_range[0], high=interact_features_range[1], shape=(interact_features_size,), dtype=float)
        
        # #TODO: are we sure about that part?
        # max_num_nodes = max(self.env.MAX_NUM_KW, self.env.MAX_NUM_DOCS)
        # kw_indices_space = spaces.Box(low=0, high=self.env.MAX_NUM_KW, shape=(2,), dtype=int)
        # doc_indices_space = spaces.Box(low=0, high=self.env.MAX_NUM_DOCS, shape=(2,), dtype=int)
        # kw2doc_indices_space = spaces.Box(low=0, high=max_num_nodes, shape=(2,), dtype=int)
        # doc2kw_indices_space = spaces.Box(low=0, high=max_num_nodes, shape=(2,), dtype=int)
        # nodes_indices_dict = {
        #     "kw":spaces.Sequence(space=kw_indices_space),
        #     "doc":spaces.Sequence(space=doc_indices_space),
        # }
        # edge_indices_dict = {
        #     "kw2doc":spaces.Sequence(space=kw2doc_indices_space),
        #     "doc2kw":spaces.Sequence(space=doc2kw_indices_space)
        # }
        # dict_space = {
        #     "X_nodes":spaces.Sequence(space=node_features_space),
        #     "X_interact":spaces.Sequence(space=interaction_feature_space),
        #     "X_nodes_indices":spaces.Dict(spaces=nodes_indices_dict),
        #     "edge_indices":spaces.Dict(spaces=edge_indices_dict),
        # }
        # self.observation_space = spaces.Dict(spaces=dict_space)

        self.observation_space = spaces.Dict()
        
        dict_space = {""}
    

    def get_obs(self):
        X_feedback = self.env.student.get_feedback_features(time_mode=self.env.time_mode, current_timestep=self.env.step_count, to_tensor=True)
        if self.type=='bassen':
            X_feedback = ObservationManager.reshape_feedback_tensor(X_feedback)
        Y_labels   = None
        data_dict = dict(corpus_graph=self.env.corpus_graph, X_feedback=X_feedback, Y_labels=Y_labels,
                    kw_normalization=None, kw_features_mean=None, kw_features_std=None,
                    clean_data_from_corpus_graphs=False)
        data_obj = SimpleNamespace(**data_dict)
        # data = CorpusGraphDataset.build_hetero_data(corpus_graph=self.env.corpus_graph, X_feedback=X_feedback, Y_labels=Y_labels,
        #                                             kw_normalization=None, kw_features_mean=None, kw_features_std=None,
        #                                             clean_data_from_corpus_graphs=False)
        return data_obj
    

    @staticmethod
    def reshape_feedback_tensor(x_feedback:torch.Tensor):
        if x_feedback.dim() == 2:  # Shape [22, 5]
            return x_feedback.view(1, -1)  # Flatten to [1, a*b]
        elif x_feedback.dim() == 3:  # Shape [batch_size, 22, 5]
            batch_size = x_feedback.size(0)
            return x_feedback.view(batch_size, -1)  # Flatten to [batch_size, 22*5]
        else:
            raise ValueError(f"Tensor of shape {x_feedback.shape} not supported.")
        
    

    # def get_obs(self):
    #     obs_object = self.build_obs_object()
    #     return obs_object.to_vector()


    # def build_obs_object(self) -> Observation:
    #     kw_features         = self.env.corpus_graph.kw_features
    #     doc_features        = self.env.corpus_graph.doc_features
    #     feedback_features_dict = self.env.student.get_feedback_features(time_mode=self.env.time_mode, current_timestep=self.env.step_count)
    #     interact_features   = torch.stack([feedback_features_dict[doc.id] for doc in self.env.corpus_graph.doc_list], dim=0)
    #     edge_indices_doc2kw = self.env.corpus_graph.doc2kw_edge_idx
    #     # edge_indices_kw2doc = self.env.corpus_graph.kw2doc_edge_idx
    #     obs_object = Observation(env=self.env,
    #                              kw_features=kw_features,
    #                              doc_features=doc_features,
    #                              interact_features=interact_features,
    #                              edge_indices_doc2kw=edge_indices_doc2kw)
    #                             #  edge_indices_kw2doc=edge_indices_kw2doc)
    #     return obs_object
    

    # @staticmethod
    # def preprocess_batches(observations:List[Observation]):
    #     X_nodes    = torch.cat([obs.get_node_features()           for obs in observations])
    #     X_feedback = torch.cat([obs.get_feedback_features()       for obs in observations])
    #     X_time     = torch.cat([obs.get_remaining_time_repeated() for obs in observations])
        
    #     nb_nodes_lengths = [0]+[obs.num_nodes for obs in observations]
    #     cum_sum_nodes    = list(np.cumsum(nb_nodes_lengths))
        
    #     kw_ranges  = [[cum_sum_nodes[i-1]                        , cum_sum_nodes[i-1]+observations[i-1].nb_kw] for i in range(1,len(cum_sum_nodes))]
    #     doc_ranges = [[cum_sum_nodes[i-1]+observations[i-1].nb_kw, cum_sum_nodes[i-1]+observations[i-1].num_nodes] for i in range(1,len(cum_sum_nodes))]
    #     nb_edges   = [ obs.num_edge_indices for obs in observations]

    #     X_nodes_indices_slices = {}
    #     X_nodes_indices_slices["kw"]  = [slice(kw_range[0] , kw_range[1] ) for kw_range  in kw_ranges]
    #     X_nodes_indices_slices["doc"] = [slice(doc_range[0], doc_range[1]) for doc_range in doc_ranges]

    #     X_nodes_indices_tensors = {}
    #     X_nodes_indices_tensors["kw"]  = torch.cat([torch.tensor(range(kw_range[0] , kw_range[1] )) for kw_range  in kw_ranges ])
    #     X_nodes_indices_tensors["doc"] = torch.cat([torch.tensor(range(doc_range[0], doc_range[1])) for doc_range in doc_ranges])
        
    #     edge_indices = {}
    #     kw_extras  = torch.cat([ torch.cat([kw_ranges[i][0]*torch.ones((1,nb_edges[i])), doc_ranges[i][0]*torch.ones((1,nb_edges[i]))],dim=0) for i in range(len(observations))], dim=1)
    #     doc_extras = torch.cat([ torch.cat([doc_ranges[i][0]*torch.ones((1,nb_edges[i])), kw_ranges[i][0]*torch.ones((1,nb_edges[i]))],dim=0) for i in range(len(observations))], dim=1)
    #     edge_indices["kw2doc"] = (torch.cat([obs.get_edge_indices_kw2doc() for obs in observations], dim=1) + kw_extras ).long()
    #     edge_indices["doc2kw"] = (torch.cat([obs.get_edge_indices_doc2kw() for obs in observations], dim=1) + doc_extras).long()
        
    #     return X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices, X_nodes_indices_slices