from genericpath import isfile
import os
from typing import Union, List, Dict, Tuple

import random
import math
import torch

# from torch.utils.data import Dataset, DataLoader

from torch.nn import Linear
import torch.nn.functional as F

# from torch.utils.data import Dataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset
from torch_geometric.data.dataset import _get_flattened_data_list
import torch_geometric.transforms as T

from ..path_manager import PathManager
from .corpus_graph import CorpusGraph
from .population import Population
from .types import Feedback, Interaction

from .. import utils



class CorpusGraphDataset(Dataset):

    DEFAULT_DATASET_CORPORA     = dict(train=["corpus1", "corpus2", "corpus4", "corpus5"], eval=["corpus3", "corpus6"])
    DEFAULT_DATASET_CORPORA2    = dict(train=["corpus1", "corpus3", "corpus4", "corpus6"], eval=["corpus2", "corpus5"])
    DEFAULT_DATASET_CORPORA3    = dict(train=["corpus2", "corpus3", "corpus5", "corpus6"], eval=["corpus1", "corpus4"])
    BIG_DATASET_CORPORA  = dict(train=["corpus1", "corpus2", "corpus4", "corpus5", "Artificial_Intelligence", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks", "ML_Zero_to_Hero", "3Blue1Brown-_Calculus", "Calculus"], eval=["corpus3", "corpus6", "Natural_Language_Processing_(NLP)_Zero_to_Hero", "Building_recommendation_systems_with_TensorFlow", "Statistics"])
    BIG_DATASET_CORPORA2 = dict(train=["corpus1", "corpus3", "corpus4", "corpus6", "Artificial_Intelligence", "Natural_Language_Processing_(NLP)_Zero_to_Hero", "Building_recommendation_systems_with_TensorFlow", "3Blue1Brown-_Calculus", "Statistics"], eval=["corpus2", "corpus5", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks", "ML_Zero_to_Hero", "Calculus"])
    BIG_DATASET_CORPORA3 = dict(train=["corpus2", "corpus3", "corpus5", "corpus6", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks", "ML_Zero_to_Hero", "Building_recommendation_systems_with_TensorFlow", "Statistics", "Calculus"], eval=["corpus1", "corpus4", "Artificial_Intelligence", "Natural_Language_Processing_(NLP)_Zero_to_Hero", "3Blue1Brown-_Calculus"])
    # BIG_DATASET_CORPORA  = dict(train=["corpus1", "corpus2", "corpus4", "corpus5", "Artificial_Intelligence", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks", "ML_Zero_to_Hero", "3Blue1Brown-_Calculus", "Calculus", "Building_recommendation_systems_with_TensorFlow"], eval=["corpus3", "corpus6", "Natural_Language_Processing_(NLP)_Zero_to_Hero", "Statistics"])
    # BIG_DATASET_CORPORA2 = dict(train=["corpus1", "corpus3", "corpus4", "corpus6", "Artificial_Intelligence", "Natural_Language_Processing_(NLP)_Zero_to_Hero", "Building_recommendation_systems_with_TensorFlow", "3Blue1Brown-_Calculus", "Statistics", "ML_Zero_to_Hero", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks"], eval=["corpus2", "corpus5",  "Calculus"])
    # BIG_DATASET_CORPORA3 = dict(train=["corpus2", "corpus3", "corpus5", "corpus6", "Building_recommendation_systems_with_TensorFlow", "Statistics", "Calculus", "Natural_Language_Processing_(NLP)_Zero_to_Hero", "ML_Zero_to_Hero", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks"], eval=["corpus1", "corpus4", "Artificial_Intelligence", , "3Blue1Brown-_Calculus"])
    FULL_TRAIN_DATASET_CORPORA1 = dict(train=["corpus1","corpus2","corpus3","corpus4","corpus5","corpus6"], eval=[])
    TINY_DATASET_CORPORA        = dict(train=["corpus1"], eval=["corpus2"])
    # LARGE_DATASET_CORPORA   = dict(train=["corpus1", "corpus2", "corpus4", "corpus5"], eval=["corpus3", "corpus6"])
    DATASET_MODE2SPLIT = {"default1": DEFAULT_DATASET_CORPORA,
                          "default2": DEFAULT_DATASET_CORPORA2,
                          "default3": DEFAULT_DATASET_CORPORA3,
                          "big1":BIG_DATASET_CORPORA,
                          "big2":BIG_DATASET_CORPORA2,
                          "big3":BIG_DATASET_CORPORA3,
                          "default":DEFAULT_DATASET_CORPORA,
                          "full_train1":FULL_TRAIN_DATASET_CORPORA1,
                          "tiny":TINY_DATASET_CORPORA,
                          "corpus1":dict(train=["corpus1"],eval=[]), "corpus2":dict(train=["corpus2"],eval=[]), "corpus3":dict(train=["corpus3"],eval=[]), "corpus4":dict(train=["corpus4"],eval=[]), "corpus5":dict(train=["corpus5"],eval=[]), "corpus6":dict(train=["corpus6"],eval=[]),
                          }
    
    def __init__(self, corpus_list:List[CorpusGraph], dataset_name:str, kw_normalization:str, is_eval=False, verbose=0):
        self.corpus_list      = corpus_list
        self.dataset_name     = dataset_name
        self.kw_normalization = kw_normalization
        self.is_eval          = is_eval
        self.verbose          = verbose

        self.idx_range2corpus_graph, self.length = self.build_idx_range2corpus()
        self.kw_features_mean, self.kw_features_std = CorpusGraphDataset.maybe_compute_dataset_stats(corpus_list=self.corpus_list, kw_normalization=self.kw_normalization, is_eval=self.is_eval, verbose=self.verbose)
        
        super().__init__(root=None, transform=None, pre_transform=None)


    @staticmethod
    def get_class_weights(loader:DataLoader, mode:str, dataset_name:str):
        loaded_class_weights = CorpusGraphDataset.load_class_weights()
        if (dataset_name in loaded_class_weights.keys()) and (mode in loaded_class_weights[dataset_name].keys()) and (loaded_class_weights[dataset_name][mode] is not None):
            class_weights = loaded_class_weights[dataset_name][mode]
            class_weights = {int(key):val for key,val in class_weights.items()}
        else:
            print("assessing class weights...")
            class_occurrences = {}
            total_samples = 0
            for data_batch in loader:  # Iterate in batches over the training/test dataset.
                values = data_batch["doc"].y
                counts = torch.bincount(values)
                for idx in range(len(counts)):
                    if not idx in class_occurrences.keys(): class_occurrences[idx] = 0
                    class_occurrences[idx] += int(counts[idx])
                total_samples += data_batch["doc"].y.shape[0]
            class_weights = {class_id: total_samples / count for class_id, count in class_occurrences.items()} # Compute weights: inverse of class frequency
            CorpusGraphDataset.save_class_weights(class_weights=class_weights, mode=mode, dataset_name=dataset_name)
            print("done")
        weights_tensor = torch.tensor([class_weights[class_idx] for class_idx in sorted(class_weights.keys())])
        return weights_tensor
    
    @staticmethod
    def load_class_weights():
        full_class_weights = utils.load_json_file(os.path.join(PathManager.STUDENT_SIMUL, "class_weights.json"))
        return full_class_weights
    @staticmethod
    def save_class_weights(class_weights:dict, mode:str, dataset_name:str):
        full_class_weights = CorpusGraphDataset.load_class_weights()
        if not dataset_name in full_class_weights.keys():
            full_class_weights[dataset_name] = {}
        full_class_weights[dataset_name][mode] = class_weights
        utils.save_dict_as_json(full_class_weights, file_path=os.path.join(PathManager.STUDENT_SIMUL, "class_weights.json"))

    
    def build_idx_range2corpus(self) -> Tuple[Dict[tuple,CorpusGraph],int]:
        idx_range2corpus_graph = {}
        current_num = 0
        for corpus_graph in self.corpus_list:
            num_xy_pairs = corpus_graph.num_reco_feedback_pairs
            new_tuple = (current_num, current_num+num_xy_pairs)
            idx_range2corpus_graph[new_tuple] = corpus_graph
            current_num += num_xy_pairs
        return idx_range2corpus_graph, current_num
    
    
    @staticmethod
    def generate_train_eval_corpus_graphs(dataset_name:str=None, split_mode:str="tiny"):
        corpus_types = ["linear"]
        if split_mode in CorpusGraphDataset.DATASET_MODE2SPLIT.keys():
            if split_mode in {"big1","big2","big3"}:
                corpus_groups = ['hand_made', 'new_corpora']
            else:
                corpus_groups = ["hand_made"]
            split_dict = CorpusGraphDataset.DATASET_MODE2SPLIT[split_mode]
            train_names, eval_names = split_dict["train"], split_dict["eval"]
        elif split_mode=='full_train2':
            corpus_groups = ['hand_made','new_corpora']
            train_names = 'all'
            eval_names = None
        else:
            raise ValueError(f'Generation mode "{split_mode}" not supported.')
        corpus_graph_list = CorpusGraph.generate_corpus_graph_list(corpus_types=corpus_types,
                                                                   corpus_groups=corpus_groups,
                                                                   feedback_pairs_folder=dataset_name,
                                                                   load_minimal_data=True)
        assert len(corpus_graph_list) > 0
        if train_names=='all':
            return corpus_graph_list, []
        else:
            train_list = [corpus_graph for corpus_graph in corpus_graph_list if corpus_graph.corpus_name in train_names]
            eval_list  = [corpus_graph for corpus_graph in corpus_graph_list if corpus_graph.corpus_name in eval_names ]
            return train_list, eval_list
    

    @staticmethod
    def generate_train_rl_corpus_graphs(training_mode:str,
                                        kw_normalization:str,
                                        split_mode:str=None,
                                        pretrain_split_mode:str=None,
                                        verbose=4):
        if training_mode=='non_linear_finetuning':
            corpus_types, corpus_groups, corpus_names  = "non_linear", "hand_made", "intro_to_ml"
            corpus_graph_list = CorpusGraph.generate_corpus_graph_list(corpus_types=corpus_types,
                                                                        corpus_groups=corpus_groups,
                                                                        corpus_names=corpus_names,
                                                                        feedback_pairs_folder=None,
                                                                        load_minimal_data=True)
        elif training_mode=='pretraining':
            assert (split_mode is not None) and split_mode != "none"
            train_list, eval_list = CorpusGraphDataset.generate_train_eval_corpus_graphs(dataset_name=None, split_mode=split_mode)
            corpus_graph_list = train_list
        else:
            raise ValueError(f'Training mode {training_mode} not supported.')
        
        assert len(corpus_graph_list) > 0

        # normalize data
        kw_features_mean, kw_features_std = CorpusGraphDataset.maybe_compute_dataset_stats(corpus_list=corpus_graph_list,
                                                                                           kw_normalization=kw_normalization,
                                                                                           is_eval=False,
                                                                                           pretrain_split_mode=pretrain_split_mode,
                                                                                           verbose=verbose)
        for corpus_graph in corpus_graph_list:
            corpus_graph.get_features_data(kw_normalization=kw_normalization, kw_features_mean=kw_features_mean, kw_features_std=kw_features_std)
        
        return corpus_graph_list
    
    
    @staticmethod
    def get_right_reco_and_check_consistency(good_recos_ids, too_hard_recos_ids, too_hard_recos_ids_not_interacted):
        assert too_hard_recos_ids_not_interacted == too_hard_recos_ids
        if len(good_recos_ids) == 1:
            right_reco = good_recos_ids[0]
        elif len(good_recos_ids) > 1:
            raise Exception("Corpus Dataset with multiple optimal recommendations not supported.")
        else:
            raise Exception("Inconsistent: no right recommendation despite the student has not completed corpus.")
        return right_reco

        
    @staticmethod
    def build_dataset(population:Population,
                      time_mode:str,
                      feedback_pairs_folder:str,
                      corpus_types:Union[str,List[str]],
                      corpus_groups:Union[str,List[str]],
                      corpus_names:Union[str,List[str]]="all",
                      MAX_NUM_NOISE_INTERACTIONS:int=4,
                      NUM_SAMPLES:int=3,
                      expert_policy:bool=False):
        assert time_mode in Interaction.TIME_MODES

        PROP_NOISE_TOO_EASY = 0.5
        PROP_NOISE_TOO_HARD = 0.5
        
        assert PROP_NOISE_TOO_EASY<1 and PROP_NOISE_TOO_HARD<1

        MAX_NUM_NOISE_SAMPLES = 40
        
        corpus_graph_list = CorpusGraph.generate_corpus_graph_list(corpus_types=corpus_types,
                                                                   corpus_groups=corpus_groups,
                                                                   corpus_names=corpus_names,
                                                                   feedback_pairs_folder=feedback_pairs_folder,
                                                                   load_minimal_data=False)
        
        for corpus_graph in corpus_graph_list:
            print(f'\nBuild dataset for corpus "{corpus_graph.corpus_name}"\n')
            
            students = population.sample_students(corpus_graph=corpus_graph, num="all")
            for student in students:
                timestep = 0
                while not student.has_completed_corpus():
                    next_interactions = student.look_ahead_next_interactions(timestep=timestep)
                    good_recos_ids = [doc_id for doc_id, interaction in next_interactions.items() if interaction.has_learned()]
                    bad_recos_ids  = [doc_id for doc_id, interaction in next_interactions.items() if not interaction.has_learned()]

                    too_easy_recos_ids = [doc_id for doc_id, interaction in next_interactions.items() if interaction.get_label()==Feedback.TOO_EASY]
                    too_hard_recos_ids = [doc_id for doc_id, interaction in next_interactions.items() if interaction.get_label()==Feedback.DO_NOT_UNDERSTAND]

                    too_easy_recos_ids_not_interacted = [doc_id for doc_id in too_easy_recos_ids if not student.has_interacted_with(doc_id=doc_id)]
                    too_hard_recos_ids_not_interacted = [doc_id for doc_id in too_hard_recos_ids if not student.has_interacted_with(doc_id=doc_id)]

                    right_reco = CorpusGraphDataset.get_right_reco_and_check_consistency(good_recos_ids, too_hard_recos_ids, too_hard_recos_ids_not_interacted)
                    
                    max_num_noise_too_easy = math.ceil(len(too_easy_recos_ids_not_interacted)*PROP_NOISE_TOO_EASY)
                    max_num_noise_too_hard = math.ceil(len(too_hard_recos_ids_not_interacted)*PROP_NOISE_TOO_HARD)
                    
                    num_not_interacted = max_num_noise_too_easy+max_num_noise_too_hard
                    num_noise_samples = min(MAX_NUM_NOISE_SAMPLES, 2**num_not_interacted-1)

                    # generate noise interactions
                    chosen_noise_ids_list = [set()]
                    for sample_idx in range(num_noise_samples):
                        chosen_noise_ids = CorpusGraphDataset.sample_interactions(max_num_noise_too_easy, max_num_noise_too_hard, too_easy_recos_ids_not_interacted, too_hard_recos_ids_not_interacted)
                        iter = 0
                        while chosen_noise_ids in chosen_noise_ids_list:
                            iter += 1
                            chosen_noise_ids = CorpusGraphDataset.sample_interactions(max_num_noise_too_easy, max_num_noise_too_hard, too_easy_recos_ids_not_interacted, too_hard_recos_ids_not_interacted)
                            assert iter <= 20
                        chosen_noise_ids_list.append(chosen_noise_ids.copy())
                    
                    # make duplicate students have these noise interactions
                    for chosen_noise_ids in chosen_noise_ids_list:
                        noise_student = student.duplicate()
                        t=timestep
                        doc_ids_list = list(chosen_noise_ids)
                        random.shuffle(doc_ids_list)
                        for doc_id in doc_ids_list:
                            noise_student.interact_with_doc_id(doc_id=doc_id, time_step=t)
                            t += 1
                        X_feedback_tensor = noise_student.get_feedback_features(time_mode=time_mode, current_timestep=t-1, to_tensor=True)
                        Y_tensor          = noise_student.look_ahead_next_interactions(timestep=t-1, to_labels=True, get_expert_policy=expert_policy)
                        corpus_graph.save_reco_feedback_pairs(X_past_interactions=X_feedback_tensor, Y=Y_tensor)
                    

                    student.interact_with_doc_id(doc_id=right_reco, time_step=timestep)
                    timestep += 1

                    ###### OLD VERSION
                    # # the maximum number of noise interactions that we add to the current stage
                    # # we need to make this depend on the number of docs we did not interact with
                    # max_num_noise_interactions = min(corpus_graph.num_docs-1, MAX_NUM_NOISE_INTERACTIONS)

                    # # num_too_easy_noise_interactions, num_too_hard_noise_interactions
                    # # number of extra noise interactions
                    # for num_noise_interactions in range(max_num_noise_interactions+1):
                    #     chosen_recos_ids_list = []
                        
                    #     # num of samples for noise interactions
                    #     num_samples = 1 if num_noise_interactions==0 else min(corpus_graph.num_docs-num_noise_interactions, NUM_SAMPLES)
                        
                    #     for idx in range(num_samples):
                    #         noise_student = student.duplicate()
                            
                    #         # add noise interactions
                    #         chosen_noise_recos_ids = set(random.sample(bad_recos_ids, num_noise_interactions))
                    #         iter_count = 0
                    #         while chosen_noise_recos_ids in chosen_recos_ids_list:
                    #             iter_count += 1
                    #             chosen_noise_recos_ids = set(random.sample(bad_recos_ids, num_noise_interactions))
                    #             assert iter_count <= 20
                    #         chosen_recos_ids_list.append(chosen_noise_recos_ids)
                            
                    #         t=timestep
                    #         for doc_id in chosen_noise_recos_ids:
                    #             noise_student.interact_with_doc_id(doc_id=doc_id, time_step=t)
                    #             t += 1
                            
                    #         X_feedback_tensor = noise_student.get_feedback_features(time_mode=time_mode, current_timestep=t-1, to_tensor=True)
                    #         Y_tensor = noise_student.look_ahead_next_interactions(timestep=t-1, to_labels=True, get_expert_policy=expert_policy)

                    #         corpus_graph.save_reco_feedback_pairs(X_past_interactions=X_feedback_tensor, Y=Y_tensor)
                    
                    # student.interact_with_doc_id(doc_id=right_reco, time_step=timestep)
                    # timestep += 1
    

    @staticmethod
    def sample_interactions(max_num_noise_too_easy, max_num_noise_too_hard, too_easy_recos_ids_not_interacted, too_hard_recos_ids_not_interacted):
        num_too_easy = random.choice(range(max_num_noise_too_easy + 1))
        num_too_hard = random.choice(range(max_num_noise_too_hard + 1))
        chosen_too_easy  = set(random.sample(too_easy_recos_ids_not_interacted, num_too_easy))
        chosen_too_hard  = set(random.sample(too_hard_recos_ids_not_interacted, num_too_hard))
        chosen_noise_ids = chosen_too_easy.union(chosen_too_hard)
        return chosen_noise_ids

    
    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data_list = _get_flattened_data_list([data for data in self])
        y = torch.cat([data["doc"].y for data in data_list if 'y' in data], dim=0)
        
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(y)

    def len(self):
        return self.length

    
    def get(self, idx):
        for idx_range, corpus_graph in self.idx_range2corpus_graph.items():
            if idx_range[0] <= idx < idx_range[1]:
                break
        data_idx = idx-idx_range[0]
        X_feedback, Y_labels = corpus_graph.load_reco_feadback_data(idx=data_idx)
        data = CorpusGraphDataset.build_hetero_data(corpus_graph=corpus_graph, X_feedback=X_feedback, Y_labels=Y_labels,
                                                    kw_normalization=self.kw_normalization, kw_features_mean=self.kw_features_mean, kw_features_std=self.kw_features_std,
                                                    clean_data_from_corpus_graphs=True)
        # data = T.NormalizeFeatures()(data)
        return data
    

    @staticmethod
    def build_hetero_data(corpus_graph:CorpusGraph, X_feedback:torch.Tensor, Y_labels:torch.Tensor=None,
                          kw_normalization:str=None, kw_features_mean:torch.Tensor=None, kw_features_std:torch.Tensor=None,
                          clean_data_from_corpus_graphs=False) -> HeteroData:
        X_nodes, kw_indices, doc_indices = corpus_graph.get_features_data(kw_normalization=kw_normalization,
                                                                          kw_features_mean=kw_features_mean,
                                                                          kw_features_std=kw_features_std)
        if clean_data_from_corpus_graphs:
            if kw_normalization is None:
                raise Exception(f'Warning: This pipeline cleans features data from corpus graphs objects without providing a way to normalize: please make sure that no normalization is required and ignore this exception.')
            corpus_graph.clean_features_data()
        
        data = HeteroData()
        kw_features = X_nodes[kw_indices]

        data["kw"].x       = kw_features
        data["doc"].x      = X_nodes[doc_indices]
        data["feedback"].x = X_feedback
        
        if Y_labels is not None:
            data["doc"].y = Y_labels
        
        data["doc", "to", "kw"].edge_index     = torch.clone(corpus_graph.doc2kw_edge_idx)
        data["kw", "to2", "doc"].edge_index    = torch.clone(corpus_graph.kw2doc_edge_idx)
        data["kw", "to_all", "doc"].edge_index = torch.clone(corpus_graph.kw2all_doc_edge_idx)

        # infos
        data["num_docs"] = corpus_graph.num_docs
        data["num_kw"]   = corpus_graph.num_kw

        data["corpus_graph"] = corpus_graph

        return data
    
    
    @staticmethod
    def maybe_compute_dataset_stats(corpus_list:List[CorpusGraph],
                                    kw_normalization:str,
                                    is_eval:bool,
                                    pretrain_split_mode:str=None,
                                    verbose:int=0):
        if kw_normalization != "z-score":
            return None, None
        else:
            dataset_stats_dir = PathManager.DATASET_STATS

            # look for the right split mode folder
            if (pretrain_split_mode is not None) and (pretrain_split_mode != 'none'):
                right_stat_folder_name = os.path.join(dataset_stats_dir, pretrain_split_mode)
            else:
                corpora_names = [corpus.corpus_name for corpus in corpus_list]
                right_stat_folder_name = None
                for folder_name in os.listdir(dataset_stats_dir):
                    folder_path = os.path.join(dataset_stats_dir, folder_name)
                    corpora_path = os.path.join(folder_path, "corpora_list.json")
                    corpora_dict = utils.load_json_file(corpora_path)
                    train_names = set(corpora_dict['train'])
                    if set(corpora_names)==train_names:
                        right_stat_folder_name = folder_path
                        break
                    eval_names  = set(corpora_dict['eval'])
                    if set(corpora_names)==eval_names and is_eval:
                        right_stat_folder_name = folder_path
                        break
            
            if right_stat_folder_name is not None:
                mean, std = CorpusGraphDataset.load_mean_std(folder_path=right_stat_folder_name)
                if (mean is not None) and (std is not None):
                    print("\nLoad dataset statistics")
                    if verbose >= 4: print(f'Mean:\n{mean}')
                    if verbose >= 4: print(f'Std:\n{std}')
                    return mean, std
            
            if is_eval:
                raise Exception(f'Required stats for z-score not found for corpora {corpora_names}')
            else:
                for mode, split_dict in CorpusGraphDataset.DATASET_MODE2SPLIT.items():
                    if set(corpora_names)==set(split_dict["train"]):
                        mean, std = CorpusGraphDataset.build_dataset_stats(corpus_list, mode, split_dict)
                        return mean, std
                raise Exception(f'Train configuration {corpora_names} not defined.')
    
    
    
    @staticmethod
    def build_dataset_stats(corpus_list:List[CorpusGraph], mode, split_dict:dict):
        print("\nBuilding dataset statistics...")
        kw_done = set()
        cumulative_sum = 0.0
        sum_of_squares = 0.0
        count = 0
        for corpus_graph in corpus_list:
            corpus_graph.get_features_data()
            for kw in corpus_graph.kw_list:
                if not kw.name in kw_done:
                    cumulative_sum += kw.features
                    sum_of_squares += kw.features ** 2
                    count += 1
            corpus_graph.clean_features_data()
        mean = cumulative_sum / count
        std = torch.sqrt(sum_of_squares / count - mean ** 2)
        print(f'Mean:\n{mean}')
        print(f'Std:\n{std}')
        CorpusGraphDataset.save_dataset_stats(mean, std, mode, split_dict)
        print("\nDone")
        return mean, std
    
    staticmethod
    def save_dataset_stats(mean, std, mode, split_dict:dict):
        folder_path = os.path.join(PathManager.STUDENT_SIMUL, "dataset_stats", mode)
        os.makedirs(folder_path, exist_ok=True)
        mean_path = os.path.join(folder_path, 'kw_features_mean.pt')
        std_path  = os.path.join(folder_path, 'kw_features_std.pt')
        corpora_list_path = os.path.join(folder_path, 'corpora_list.json')
        torch.save(mean, f=mean_path)
        torch.save(std, f=std_path)
        utils.save_dict_as_json(data_dict=split_dict, file_path=corpora_list_path)


    @staticmethod
    def load_mean_std(folder_path:str):
        mean_file_path = os.path.join(folder_path, 'kw_features_mean.pt')
        std_file_path  = os.path.join(folder_path, 'kw_features_std.pt')
        if os.path.isfile(mean_file_path) and os.path.isfile(std_file_path):
            mean = torch.load(f=mean_file_path)
            std = torch.load(f=std_file_path)
            return mean, std
        else:
            return None, None
        



if __name__=="__main__":

    build_dataset = True
    load_dataset  = False
    
    # only for build_dataset
    corpus_types  = ["linear"]
    corpus_groups = ["hand_made", "new_corpora"]
    corpus_names  = "all"
    
    if build_dataset:
        folder_name = "expert_policy2" 
        prior_knowledge_distrib = "all"
        prereq_distrib = "all"
        population = Population(prior_knowledge_distrib=prior_knowledge_distrib,
                                prereq_distrib=prereq_distrib,
                                feedback_mode="default")
        CorpusGraphDataset.build_dataset(corpus_types=corpus_types,
                                        corpus_groups=corpus_groups,
                                        corpus_names=corpus_names,
                                        population=population,
                                        time_mode=Interaction.ONLY_LAST_TIME_MODE,
                                        feedback_pairs_folder=folder_name,
                                        expert_policy=True)
    
    elif load_dataset:
        
        dataset_name = "first_dataset"
        mode = "big3"
        hidden_dim = 32
        batch_size_train = 2
        batch_size_eval = 2
        lr = 0.01
        num_epochs = 2
        kw_normalization = "none"
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = "cpu"
        
        train_graphs, eval_graphs = CorpusGraphDataset.generate_train_eval_corpus_graphs(dataset_name=dataset_name,split_mode=mode)
        train_dataset = CorpusGraphDataset(corpus_list=train_graphs, dataset_name=dataset_name, kw_normalization=kw_normalization, is_eval=False, verbose=4)
        eval_dataset  = CorpusGraphDataset(corpus_list=eval_graphs,  dataset_name=dataset_name, kw_normalization=kw_normalization, is_eval=True, verbose=4)
        print(len(train_dataset), len(eval_dataset))
        print('ok')
        # train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        # eval_loader   = DataLoader(eval_dataset,  batch_size=batch_size_eval, shuffle=False)
        
        # # print(train_dataset.num_classes)
        # data_sample = train_dataset[0]
        # print(data_sample.metadata())

        # num_data = 1
        # count = 0
        # for data_batch in train_loader:
        #     count += 1
        #     if count > num_data:
        #         break
        #     print(f'\nBatch {count}:')
        #     print(data_batch)
        #     print()
        
        # count = 0
        # for data_batch in eval_loader:
        #     count += 1
        #     if count > num_data:
        #         break
        #     print(f'\nBatch {count}:')
        #     print(data_batch)
        #     print()