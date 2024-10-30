import os
import sys
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
import pprint
import copy

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F

from .types import AbstractRequirement, Knowledge, ProbabilisticRequirement, KW, Doc, KC, AbstractRequirementsMap, RequirementsMap
from . import utils_student_simul
from ..path_manager import PathManager

from .. import utils


class CorpusGraph:
    
    CORPUS_DATA_STRUCTURE = ["feature_type", "folder"]
    DOC2KW_FILE = "doc2kw.json"
    DOC2KC_FILE = "doc2kc.json"
    RECO_FEEDBACK_PAIRS = "reco_feedback_pairs"
    BACKGROUND_REQUIREMENTS_FILE = "background_requirements.json"

    # KW_EMBEDDING_MIN_VAL = -7
    # KW_EMBEDDING_MAX_VAL = 7
    KW_EMBEDDING_MIN_VAL = -6.9
    KW_EMBEDDING_MAX_VAL =  6.9

    
    def __init__(self,
                 corpus_name:str="corpus2", corpus_group:str="hand_made", corpus_type:str="linear",
                 kw_features_type:str="standard1",
                 feedback_pairs_folder:str=None,
                 doc_features_type:str="mean",
                 load_minimal_data:bool=False,
                 add_doc_degree:bool=False,  # WARNING! INCREASES THE CHANCE OF OVERFITTING
                 kw_normalization:str=None,
                 kw_features_mean:str=None,
                 kw_features_std:str=None):
        
        self.corpus_name    = corpus_name
        self.corpus_group   = corpus_group
        self.corpus_type    = corpus_type
        self.add_doc_degree = add_doc_degree

        self.kw_features_type = kw_features_type
        self.doc_features_type = doc_features_type

        self.is_features_set = False
        self.corpus_id = self.generate_corpus_id()
        
        self.corpus_folder_path = PathManager.GET_CORPUS_PATH(data_type="extracted_data", corpus_name=self.corpus_name, corpus_group=self.corpus_group, corpus_type=self.corpus_type)
        
        doc_name2kw_names, doc_name2kc_names, background_requirements = self.load_files_data()
        self._preprocess_data(doc_name2kw_names, doc_name2kc_names, background_requirements)
        self._set_requirements(background_requirements)

        if feedback_pairs_folder is not None:
            self.reco_feedback_pairs_path = self.get_reco_feedback_pairs_path(feedback_pairs_folder)
            self.num_reco_feedback_pairs = self.find_num_reco_feedback_pairs()

        if not load_minimal_data:
            self.get_features_data(kw_normalization=kw_normalization, kw_features_mean=kw_features_mean, kw_features_std=kw_features_std)
    

    def get_features_data(self, kw_normalization:str=None, kw_features_mean:torch.Tensor=None, kw_features_std:torch.Tensor=None) -> torch.Tensor:
        if not self.is_features_set:
            self.kw_normalization = kw_normalization
            self.set_node_features(kw_normalization=kw_normalization, kw_features_mean=kw_features_mean, kw_features_std=kw_features_std)
            self.set_features_matrices()
            self.is_features_set = True
        return torch.clone(self.features_matrix), torch.clone(self.kw_indices), torch.clone(self.doc_indices)
    

    def clean_features_data(self):
        self.clean_node_features()
        self.clean_features_matrices()
        self.features_size = None
        self.is_features_set = False
    

    def clean_node_features(self):
        for kw in self.kw_list:
            kw.features = None
        for doc in self.doc_list:
            doc.features = None
    
    def clean_features_matrices(self):
        self.features_matrix, self.kw_indices, self.doc_indices = None, None, None


    def set_node_features(self, kw_normalization:str=None, kw_features_mean:torch.Tensor=None, kw_features_std:torch.Tensor=None):
        kw_name2features = self.load_kw_features()
        if kw_normalization is None or kw_normalization !='none':
            # print('!!! WARNING: no normalization for keyword features. !!!\n')
            pass
        else:
            # print(f'Normalizing kw features with method "{kw_normalization}"')
            pass
        for kw in self.kw_list:
            kw_vec = kw_name2features[kw.name]
            kw_vec = self.maybe_normalize(vector=kw_vec,
                                          kw_normalization=kw_normalization,
                                          min_val=CorpusGraph.KW_EMBEDDING_MIN_VAL,
                                          max_val=CorpusGraph.KW_EMBEDDING_MAX_VAL,
                                          mean=kw_features_mean,
                                          std=kw_features_std)
            kw.features = kw_vec
            kw_features_size = kw.features.shape[0]
        for doc in self.doc_list:
            doc.build_features(feature_type=self.doc_features_type, add_doc_degree=self.add_doc_degree)
            doc_features_size = doc.features.shape[0]
        assert kw_features_size == doc_features_size
        self.features_size = kw_features_size
    

    def maybe_normalize(self, vector:torch.Tensor, kw_normalization:str=None, min_val:float=None, max_val:float=None, mean:torch.Tensor=None, std:torch.Tensor=None):
        if kw_normalization is None or kw_normalization=="none":
            self.features_range = (min_val, max_val)
            return vector
        elif kw_normalization == "z-score":
            normalized_vector = (vector - mean) / std
            new_min_val = min_val + float(torch.min(mean)) / float(torch.min(std))
            new_max_val = max_val + float(torch.max(mean)) / float(torch.min(std))
            self.features_range = (new_min_val, new_max_val)
            return normalized_vector
        elif kw_normalization == "unit":
            self.features_range = (-1, 1)
            return F.normalize(vector, p=2, dim=0)
        elif kw_normalization == "min_max":
            raise NotImplementedError()
            self.features_range = (0, 1)
            return self.min_max_scaling(vector=vector, min_val=min_val, max_val=max_val)
        elif kw_normalization == "standardization":
            raise NotImplementedError()
            self.features_range = ...
            return self.standardization(vector=vector)
        else:
            raise Exception(f'Normalization mode "{kw_normalization}" not supported')


    def maybe_preprocess_kw_features(self, raw_kw_features:torch.Tensor):
        if self.kw_normalization=="z-score":
            normalized_data = (raw_kw_features - self.kw_features_mean) / self.kw_features_std
            return normalized_data
        elif self.kw_normalization=="unit":
            return F.normalize(raw_kw_features, p=2, dim=1)
        elif self.kw_normalization=="none":
            return raw_kw_features
        else:
            raise ValueError(f'Normalization method {self.kw_normalization} not supported.')
    
    
    def load_files_data(self):
        doc2kw_file_path = os.path.join(self.corpus_folder_path, CorpusGraph.DOC2KW_FILE)
        doc2kc_file_path = os.path.join(self.corpus_folder_path, CorpusGraph.DOC2KC_FILE)
        background_requirements_path = os.path.join(self.corpus_folder_path, CorpusGraph.BACKGROUND_REQUIREMENTS_FILE)
        
        doc_name2kw_names = self.load_doc_name2any(doc2kw_file_path)
        doc_name2kc_names = self.load_doc_name2any(doc2kc_file_path)
        background_requirements = self.load_doc_name2any(background_requirements_path, accept_non_existence=True)

        self.check_files_consistency(doc_name2kw_names, doc_name2kc_names)
        return doc_name2kw_names, doc_name2kc_names, background_requirements

    
    def _preprocess_data(self,
                         doc_name2kw_names:Dict[str,List[str]],
                         doc_name2kc_names:Dict[str,List[str]],
                         background_requirements:Dict[str,List[str]]):
        self.doc_name2obj:Dict[str,Doc] = {}
        self.kw_name2obj:Dict[str,KW]  = {}
        self.kc_name2obj:Dict[str,KC]  = {}

        self.doc_id2obj:Dict[int,Doc] = {}
        self.kw_id2obj:Dict[int,KW]  = {}
        self.kc_id2obj:Dict[int,KC]  = {}

        self.doc_list:List[Doc] = []
        self.kw_list:List[KW]   = []
        self.kc_list:List[KC]   = []

        self.current_doc_idx = 0
        self.current_kw_idx  = 0
        self.current_kc_idx  = 0

        self.num_bipartite_edges = 0
        for doc_name in doc_name2kw_names:
            kw_names_list = doc_name2kw_names[doc_name]
            kc_names_list = doc_name2kc_names[doc_name]
            self._build_document(doc_name, kw_names_list, kc_names_list)
        
        self._build_background_kc(background_requirements)
        
        self.num_kw   = len(self.kw_list)
        self.num_docs = len(self.doc_list)
        self.num_kc   = len(self.kc_list)

        self.num_bipartite_nodes = self.num_docs+self.num_kw
        
        self._preprocess_bipartite_edges()


    def _build_document(self, doc_name:str, kw_names_list:List[str], kc_names_list:List[str]):
        self.check_no_duplicates(doc_name, kw_names_list, kc_names_list)
        kw_list = self._build_kw_list(kw_names_list=kw_names_list)
        kc_list = self._build_kc_list(kc_names_list)
        doc_id = self.current_doc_idx
        doc = Doc(name=doc_name, id=doc_id, kw_list=kw_list, kc_list=kc_list)
        # doc.build_features(feature_type=self.doc_features_type, add_doc_degree=self.add_doc_degree)

        self.doc_id2obj[doc_id] = doc
        self.doc_name2obj[doc_name] = doc
        self.doc_list.append(doc)
        self.current_doc_idx += 1
        self.num_bipartite_edges += len(kw_list)
    

    def _build_background_kc(self, background_requirements:Dict[str,List[str]]):
        if background_requirements is not None:
            current_id = max([kc.id for kc in self.kc_list]) + 1
            for kc_name, kc_requirements_names in background_requirements.items():
                if not kc_name in self.kc_name2obj.keys():
                    raise Exception(f'Knowledge component "{kc_name}" has background requirements in "{self.BACKGROUND_REQUIREMENTS_FILE}" file but is not mentioned in {self.DOC2KC_FILE} file.')
                for kc_requirement_name in kc_requirements_names:
                    new_id = current_id
                    kc = KC(name=kc_requirement_name, id=new_id, type=KC.BACKGROUND_TYPE)
                    self.kc_id2obj[new_id] = kc
                    self.kc_name2obj[kc_requirement_name] = kc
                    self.kc_list.append(kc)
                    current_id += 1
            
    

    def _build_kw_list(self, kw_names_list:List[str]):
        kw_list = []
        for kw_name in kw_names_list:
            if kw_name in self.kw_name2obj.keys():
                kw = self.kw_name2obj[kw_name]
            else:
                kw_id = self.current_kw_idx
                kw = KW(name=kw_name, id=kw_id, features=None)
                self.kw_id2obj[kw_id] = kw
                self.kw_name2obj[kw_name] = kw
                self.kw_list.append(kw)
                self.current_kw_idx += 1
            kw_list.append(kw)
        return kw_list
    

    def _build_kc_list(self, kc_names_list):
        kc_list = []
        for kc_name in kc_names_list:
            if kc_name in self.kc_name2obj.keys():
                kc = self.kc_name2obj[kc_name]
            else:
                kc_id = self.current_kc_idx
                kc = KC(name=kc_name, id=kc_id)
                self.kc_id2obj[kc_id]     = kc
                self.kc_name2obj[kc_name] = kc
                self.kc_list.append(kc)
                self.current_kc_idx += 1
            kc_list.append(kc)
        return kc_list
    

    def min_max_scaling(self, vector:torch.Tensor, min_val=-7, max_val=7):
        normalized_tensor = (vector - min_val) / (max_val - min_val)
        return normalized_tensor

    def standardization(self, vector:torch.Tensor):
        raise NotImplementedError()
    

    def _set_requirements(self, background_requirements:Dict[str,List[str]]):
        abstract_requirements_dict = {}
        for kc_id, kc in self.kc_id2obj.items():
                abstract_requirements_dict[kc_id] = self.load_kc_requirements(kc)
                if background_requirements is not None:
                    if kc.name in background_requirements.keys():
                            background_required_kc_list = [ self.kc_name2obj[req_kc_name] for req_kc_name in background_requirements[kc.name]]
                            background_requirements_list = [ AbstractRequirement(kc=req_kc, type="absolute") for req_kc in background_required_kc_list]
                            abstract_requirements_dict[kc_id] += background_requirements_list
        self.abstract_requirements_map = AbstractRequirementsMap(kc_id2obj=self.kc_id2obj, map=abstract_requirements_dict)
    

    def load_kc_requirements(self, kc:KC):
        if self.corpus_type == PathManager.LINEAR_CORPUS_TYPE:
            requirements = self.decode_linear_requirements(kc=kc)
        elif self.corpus_type == PathManager.NON_LINEAR_CORPUS_TYPE:
            requirements = self.decode_non_linear_requirements(kc=kc)
        else:
            raise Exception(f"Corpus type {self.corpus_type} not supported.")
        return requirements
    

    def decode_linear_requirements(self, kc:KC) -> List[AbstractRequirement]:
        if kc.name == "0":
            requirements = []
        elif kc.name == "1" and all(kc.name != "0" for kc in self.kc_list):
            requirements = []
        else:
            prereq_kc_name = str(int(kc.name)-1)
            prereq_kc = self.kc_name2obj[prereq_kc_name]
            requirements = [AbstractRequirement(kc=prereq_kc, type="absolute")]
        return requirements

    
    def decode_non_linear_requirements(self, kc:KC) -> List[AbstractRequirement]:
        kc_name = kc.name
        requirements = []
        if kc.type == "background":
            requirements = []
        elif kc.type == "regular":
            level_idx_str, sublevel_idx_str = kc_name.split("_")
            level_idx, sublevel_idx = int(level_idx_str), int(sublevel_idx_str)
            if sublevel_idx > 0:
                learning_pref_kc_name = f"{level_idx}_{sublevel_idx-1}"
                kc = self.kc_name2obj[learning_pref_kc_name]
                learning_prefs = [AbstractRequirement(kc=kc, type="settable")]
                requirements += learning_prefs
            if level_idx > 0 and sublevel_idx > 0:
                learning_pref_kc_name = f"{level_idx-1}_{sublevel_idx}"
                kc = self.kc_name2obj[learning_pref_kc_name]
                mandatory_requirements = [ AbstractRequirement(kc=kc, type="absolute") ]
                requirements += mandatory_requirements
        return requirements


    def check_files_consistency(self, doc_name2kw_names:dict, doc_name2kc_names:dict):
        assert doc_name2kw_names.keys() == doc_name2kc_names.keys()


    def set_features_matrices(self):
        kw_features_matrix  = torch.cat([ torch.unsqueeze(kw.features , dim=0) for kw_id, kw   in self.kw_id2obj.items() ], dim=0)
        doc_features_matrix = torch.cat([ torch.unsqueeze(doc.features, dim=0) for doc_id, doc in self.doc_id2obj.items()], dim=0)
        
        num_kw   = kw_features_matrix.shape[0]
        num_docs = doc_features_matrix.shape[0]
        kw_indices  = torch.tensor(list(range(0, num_kw)))
        doc_indices = torch.tensor(list(range(num_kw, num_kw+num_docs)))

        features_matrix = torch.cat([kw_features_matrix, doc_features_matrix], dim=0)

        self.features_matrix = features_matrix
        self.kw_indices      = kw_indices
        self.doc_indices     = doc_indices
        # return features_matrix, kw_indices, doc_indices
    

    def load_doc_name2any(self, filepath:str, accept_non_existence=False) -> dict:
        raw_doc2any = utils.load_json_file(filepath, accept_none=True)
        if raw_doc2any is None:
            if accept_non_existence:
                return None
            else:
                raise Exception(f'File "{filepath}" does not exist.')
        doc2any = { doc_str:any_list for doc_str, any_list in raw_doc2any.items()}
        return doc2any


    def check_no_duplicates(self, doc_name:str, *lists):
        for list in lists:
            has_duplicates, item = utils.has_duplicates(list)
            if has_duplicates:
                raise Exception(f'Duplicate "{item}" found in "{self.corpus_id}" data, document "{doc_name}":\n{list}')
    

    def get_regular_kc(self) -> List[KC]:
        return [kc for kc in self.kc_list if kc.is_regular()]
    
    def get_background_kc(self) -> List[KC]:
        return [kc for kc in self.kc_list if kc.is_background()]

    
    def _preprocess_bipartite_edges(self):
        # doc2kw
        self._doc2kw_edge_idx = torch.zeros((2,self.num_bipartite_edges))
        i = 0
        for doc in self.doc_list:
            for kw in doc.kw_list:
                self._doc2kw_edge_idx[0,i] = doc.id
                self._doc2kw_edge_idx[1,i] = kw.id
                i+=1
        assert i == self.num_bipartite_edges
        
        # kw2doc
        self._kw2doc_edge_idx = torch.clone(self._doc2kw_edge_idx).flip(0)
        combined = self._kw2doc_edge_idx[0] * self._kw2doc_edge_idx.max() + self._kw2doc_edge_idx[1]
        sorted_indices = combined.sort()[1]
        self._kw2doc_edge_idx = self._kw2doc_edge_idx[:, sorted_indices]
        
        # kw2all_doc
        num_fully_connected_edges = self.num_docs*self.num_kw
        self._kw2all_doc_edge_idx = torch.zeros((2,num_fully_connected_edges))
        j = 0
        for kw in self.kw_list:
            for doc in self.doc_list:
                self._kw2all_doc_edge_idx[0,j] = kw.id
                self._kw2all_doc_edge_idx[1,j] = doc.id
                j+=1
        assert j == num_fully_connected_edges


    def load_kw_features(self) -> Dict[str, torch.Tensor]:
        filename = "embeddings.npy"
        file_path = os.path.join(self.corpus_folder_path, "embedding", self.kw_features_type, filename)
        data = np.load(file_path, allow_pickle=True)
        words2vec = data[()]
        words2vec_tensors = {word_str:torch.from_numpy(vec_np) for word_str,vec_np in words2vec.items()}
        return words2vec_tensors
    

    def generate_corpus_id(self) -> str:
        id = f"{self.corpus_type}_{self.corpus_group}_{self.corpus_name}"
        return id
    

    def get_features_size_and_range(self) -> Tuple[int, Tuple[int,int]]:
        return self.features_size, self.features_range
    

    @staticmethod
    def generate_corpus_graph_list(corpus_types:Union[str,List[str]]="all",
                                   corpus_groups:Union[str,List[str]]="all",
                                   corpus_names:Union[str,List[str]]="all",
                                   feedback_pairs_folder:str=None,
                                   load_minimal_data:bool=False) -> List["CorpusGraph"]:
        corpus_graph_list = []
        for corpus_name, corpus_group, corpus_type, _, _ in PathManager.BROWSE_CORPORA(data_types="extracted_data", corpus_types=corpus_types, corpus_groups=corpus_groups, corpus_names=corpus_names):
            corpus_graph = CorpusGraph(corpus_name=corpus_name, corpus_group=corpus_group, corpus_type=corpus_type,
                                       load_minimal_data=load_minimal_data,
                                       feedback_pairs_folder=feedback_pairs_folder)
            corpus_graph_list.append(corpus_graph)
        return corpus_graph_list


    def print_main_features(self):
        print("\n### Keywords ###")
        print(f"\nkeywords ids:")
        pprint.pprint({kw.id:kw.name for kw in self.kw_list})
        print(f"\nKeywords features:")
        pprint.pprint({kw.id:kw.features for kw in self.kw_list})
        print("\n### Documents ###")
        print(f"\nDocs ids:")
        pprint.pprint({doc.id:doc.name for doc in self.doc_list})
        print(f"\nDocs features:")
        pprint.pprint({doc.id:doc.features for doc in self.doc_list})
        print("\n### Documents and Keywords ###")
        print(f"\ndoc --> kw:")
        pprint.pprint({doc.name:[kw.name for kw in doc.kw_list] for doc in self.doc_list})
    

    def build_bipartite_nx_graph(self):
        nx_graph = nx.Graph()
        doc_names_list = [ doc.name for doc in self.doc_list ]
        kw_names_list  = [ kw.name  for kw  in self.kw_list ]
        edges_list = [ (doc.name, kw.name) for doc in self.doc_list for kw in doc.kw_list ]

        nx_graph.add_nodes_from(doc_names_list, bipartite=0)
        nx_graph.add_nodes_from(kw_names_list, bipartite=1)
        nx_graph.add_edges_from(edges_list)
        return nx_graph
    

    def display_bipartite_graph(self):
        nx_graph = self.build_bipartite_nx_graph()
        kw_names_list  = [ kw.name  for kw  in self.kw_list  ]
        doc_names_list = [ doc.name for doc in self.doc_list ]
        
        visu_dict = self.get_visu_params()
        keyword_nodes_colors = [visu_dict["std_node_col"]]*self.num_kw
        node_color = [(1,1,1) for p in range(self.num_docs)] + keyword_nodes_colors
        linewidths = [3]*self.num_docs+ [1]*self.num_kw

        # edgecolors = self.state2edge_color(s_t, act=None) + [visu_dict["std_node_col"]]*self.env.nb_kw
        edgecolors = ["black"]*self.num_bipartite_nodes
        node_size  = visu_dict["node_size"]
        
        drawing_params = {"node_color":node_color, "edgecolors":edgecolors, "node_size":node_size, "linewidths":linewidths}
        pos = utils_student_simul.bipartite_layout(nx_graph, nodes=kw_names_list)
        kw_label_pos = {}
        doc_label_pos = {}
        for kw in kw_names_list:
            kw_label_pos[kw] = copy.deepcopy(pos[kw])
            kw_label_pos[kw][0] -= 0.05
        for doc in doc_names_list:
            doc_label_pos[doc] = copy.deepcopy(pos[doc])
            doc_label_pos[doc][0] += 0.05
        labels_pos = {**kw_label_pos, **doc_label_pos}
        fig = plt.figure()
        nx.draw(nx_graph, pos, with_labels=False, **drawing_params)
        nx.draw_networkx_labels(nx_graph, labels_pos, horizontalalignment="center", font_size=8)
        plt.show()
    
    
    def save_reco_feedback_pairs(self, X_past_interactions:torch.Tensor, Y:torch.Tensor):
        new_XY_folder_path = os.path.join(self.reco_feedback_pairs_path)
        if not os.path.isdir(new_XY_folder_path):
            os.mkdir(path=new_XY_folder_path)
        
        idx = self.num_reco_feedback_pairs
        filename = self.reco_feedback_idx2filename(idx=idx)
        filepath = os.path.join(new_XY_folder_path, filename)
        torch.save((X_past_interactions,Y), f=filepath)
        self.num_reco_feedback_pairs += 1
    
    def reco_feedback_idx2filename(self, idx:int):
        return f'{idx}_XY.pt'
        

    def find_num_reco_feedback_pairs(self) -> int:
        pt_file_count = 0
        for filename in os.listdir(self.reco_feedback_pairs_path):
            if filename.endswith('.pt'):
                pt_file_count += 1
        return pt_file_count
    

    def get_reco_feedback_pairs_path(self, feedback_pairs_folder):
        XY_dir_path = os.path.join(self.corpus_folder_path, self.RECO_FEEDBACK_PAIRS, feedback_pairs_folder)
        if not os.path.isdir(XY_dir_path):
            os.makedirs(name=XY_dir_path)
        return XY_dir_path
    

    def load_reco_feadback_data(self, idx:int):
        filename = self.reco_feedback_idx2filename(idx=idx)
        filepath = os.path.join(self.reco_feedback_pairs_path, filename)
        data = torch.load(f=filepath)
        return data

    
    def display_abstract_requirements_graph(self):
        nodes = self.abstract_requirements_map.get_nodes()
        settable_edges, non_settable_edges = self.abstract_requirements_map.to_edges()
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(settable_edges, p="settable")
        nx_graph.add_edges_from(non_settable_edges, p="definitive")
        pos = utils_student_simul.grid_layout(nodes_list=nodes, two_dims=self.corpus_type==PathManager.NON_LINEAR_CORPUS_TYPE)
        fig = plt.figure()
        nx.draw(nx_graph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, rotate=False)
        plt.show()

    
    def get_visu_params(self):
        visu_dict = {}
        visu_dict["figsize"]=[10, 7.5]
        visu_dict["std_node_col"] = "#1f78b4"
        visu_dict["linewidths"] = [3]*self.num_docs+ [1]*self.num_kw
        visu_dict["node_size"]  = [300]*self.num_bipartite_nodes
        visu_dict["act_size"]   = 400
        return visu_dict
    
    @property
    def doc2kw_edge_idx(self):
        return self._doc2kw_edge_idx
    
    @property
    def kw2doc_edge_idx(self):
        return self._kw2doc_edge_idx
    
    @property
    def kw2all_doc_edge_idx(self):
        return self._kw2all_doc_edge_idx
    


def get_total_size(obj, seen=None):
    """Recursively finds size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])

    return size


if __name__ == "__main__":

    import time

    t1 = time.time()
    
    # corpus_type = "non_linear"
    # corpus_name = "intro_to_ml"
    # corpus_graph = CorpusGraph(corpus_name=corpus_name, corpus_group="hand_made", corpus_type=corpus_type)
    # corpus_graph.print_main_features()
    
    # # corpus_graph.display_bipartite_graph()
    # corpus_graph.display_abstract_requirements_graph()

    corpus_graph_list = CorpusGraph.generate_corpus_graph_list(corpus_types=["linear"], corpus_groups=["hand_made"], load_minimal_data=False)
    
    t2 = time.time()

    print("Loading time: ",t2-t1)

    print("size", get_total_size(corpus_graph_list[0]))