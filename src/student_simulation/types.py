from abc import ABC

from typing import TYPE_CHECKING, List, Dict, Tuple, Union, Any

import random
import torch
import numpy as np
import pandas as pd

import pprint

if TYPE_CHECKING:
    from .student import Student


class ProbaDistrib:
    def __init__(self):
        pass
    def sample(self, discrete_space_size:int=None, num=1) -> Union[float, List[float]]:
        raise NotImplementedError()

class BinomialDistrib(ProbaDistrib):
    def __init__(self, n:int=1, mean:float=None):
        self.mean = mean
        self.n = n
    
    def sample(self, num=1, **kwargs):
        if num>1:
            return list(np.random.binomial(n=self.n, p=self.mean, size=num))
        else:
            return np.random.binomial(n=self.n, p=self.mean)

class UniformDistrib(ProbaDistrib):
    def __init__(self, min_val:float=None, max_val:float=None):
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self, num=1, **kwargs):
        if num>1:
            return [round(random.uniform(self.min_val, self.max_val), 2) for idx in range(num)]
        else:
            return round(random.uniform(self.min_val, self.max_val), 2)

class DecreasingExponential(ProbaDistrib):
    def __init__(self, lam:float):
        self.lam = lam
    
    def sample(self, discrete_space_size:int, num:int=1):
        # Calculate the probabilities for each value in the support
        probabilities = np.exp(-self.lam * np.arange(0, discrete_space_size ))
        probabilities /= probabilities.sum()
        # Sample the specified number of samples based on the calculated probabilities
        return int(np.random.choice(np.arange(0, discrete_space_size), size=num, p=probabilities))

class ZeroDistrib(ProbaDistrib):
    def __init__(self):
        pass
    def sample(self, num:int=1, **kwargs):
        return [0 for idx in range(num)]

class UniformDiscreteDistrib(ProbaDistrib):
    def __init__(self):
        pass
    def sample(self, discrete_space_size:int, num:int=1):
        return random.sample(range(discrete_space_size+1), k=num)


class KW:
    """Keyword object"""
    def __init__(self, name:str, id:int, features:torch.Tensor=None):
        self._name = name
        self._id = id
        self._features = features

        self.feature_size = self._features.shape[0] if features is not None else None
    
    @property
    def name(self):
        return self._name
    
    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, value:int):
        self._id = value
    
    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value:torch.Tensor):
        self._features = value
        self.feature_size = self._features.shape[0] if self._features is not None else None


class KC:
    """Knowledge Component object"""
    BACKGROUND_TYPE="background"
    REGULAR_TYPE = "regular"
    LEVEL2VAL = {"0":1, "1":2, "2":3}
    def __init__(self, name:str, id:int, type:str="regular"):
        self._name = name
        self._id   = id
        self.type  = type
        
        self.level = self.set_level()
        self.value = self.set_value()
    
    def set_level(self):
        if "_" in self._name:
            return self._name.split('_')[1]
        else:
            return "1"
    
    def set_value(self):
        if self.type == KC.BACKGROUND_TYPE:
            return 0
        elif "_" in self._name:
            assert self.level in KC.LEVEL2VAL.keys(), f'Sublevel "{self.level}" not in KC.LEVEL2VAL keys.'
            return KC.LEVEL2VAL[self.level]
        else:
            return 1
    
    def is_regular(self):
        return self.type==KC.REGULAR_TYPE
    
    def is_background(self):
        return self.type==KC.BACKGROUND_TYPE
    
    @property
    def name(self):
        return self._name
    
    @property
    def id(self):
        return self._id


class Doc:
    """Document object"""
    def __init__(self, name:str, id:int, kw_list:List[KW], kc_list:List[KC]=None, features:torch.Tensor=None):
        self._name = name
        self._id = id
        self._kw_list = kw_list
        self._features = features
        self.kc_list = kc_list
        
        self.num_kw = len(self._kw_list)

    def build_features(self, feature_type:str, feature_size:int=None, add_doc_degree:bool=False):
        if feature_type == "zero":
            if feature_size is None:
                feature_size = self.kw_list[0]
            self._features = torch.zeros(feature_size)
        elif feature_type == "mean":
            tensors_list = [ kw.features for kw in self._kw_list ]
            stacked_tensors = torch.stack(tensors_list)
            self._features = torch.mean(stacked_tensors, dim=0)
        else:
            raise Exception(f'Feature type "{feature_type}" not supported.')
        
        if add_doc_degree:
            num_vectors_tensor = torch.tensor([self.num_kw])
            self._features = torch.cat((self._features, num_vectors_tensor), dim=0)
    
    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id
    
    @property
    def kw_list(self):
        return self._kw_list

    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value:torch.Tensor):
        self._features = value
    


class AbstractRequirement:
    REQUIREMENT_TYPES = {"absolute", "settable"}

    def __init__(self, kc:KC, type:str):
        self.kc = kc
        self.type = type
        assert self.type in AbstractRequirement.REQUIREMENT_TYPES

        self.is_settable = self.type == "settable"
        
    def __str__(self):
        return f"AbstractRequirement(kc_id={self.kc.id}, type={self.type})"
    
    def __repr__(self):
        return f"AbstractRequirement(kc_id={self.kc.id}, type={self.type})"
    
    def generate_probabilistic_requirement(self, proba_distrib:ProbaDistrib=None):
        if self.type == "absolute":
            return ProbabilisticRequirement(kc=self.kc, proba_distrib=proba_distrib, absolute=True)
        elif self.type == "settable":
            return ProbabilisticRequirement(kc=self.kc, proba_distrib=proba_distrib, absolute=False)


class ProbabilisticRequirement:
    def __init__(self, kc:KC, absolute:bool, proba_distrib:ProbaDistrib=None):
        self.kc = kc
        self.proba_distrib = proba_distrib
        self.absolute = absolute
        if self.absolute:
            self.sample = lambda num=1: True
        else:
            self.sample = lambda num=1: self.sample_from_proba_distrib(num=num)
    
    def __str__(self):
        return f"ProbabilisticRequirement(kc_id={self.kc.id}, proba={self.proba})"
    
    def __repr__(self):
        return f"ProbabilisticRequirement(kc_id={self.kc.id}, proba={self.proba})"
    
    def sample_from_proba_distrib(self, num=1):
        return self.proba_distrib.sample(num=num)


class Requirement:
    def __init__(self, kc:KC):
        self.kc = kc


class KCMap:
    def __init__(self, kc_id2obj:Dict[int,KC], map:Dict[int,Any]):
        self.kc_id2obj = kc_id2obj
        self.map = map
    
    def get_nodes(self, node_type="name") -> list:
        if node_type == "name":
            return [self.kc_id2obj[kc_id].name for kc_id in self.kc_id2obj.keys()]
        elif node_type == "id":
            return [self.kc_id2obj[kc_id].id for kc_id in self.kc_id2obj.keys()]
        else:
            raise Exception(f'Edge type {node_type} not supported.')


class AbstractRequirementsMap(KCMap):
    def __init__(self, kc_id2obj:Dict[int,KC], map:Dict[int,List[AbstractRequirement]]):
        super().__init__(kc_id2obj, map)
        self.map: Dict[int,List[AbstractRequirement]]

    def to_edges(self, edge_type="name") -> Tuple[List[tuple],List[tuple]]:
        kc2abstract_req_tuples = [(abstract_req, self.kc_id2obj[kc_id]) for kc_id, abstract_req_list in self.map.items() for abstract_req in abstract_req_list]
        settable_edges     = [ (abstract_req.kc, kc) for abstract_req, kc in kc2abstract_req_tuples if abstract_req.is_settable]
        non_settable_edges = [ (abstract_req.kc, kc) for abstract_req, kc in kc2abstract_req_tuples if not abstract_req.is_settable]
        if edge_type == "name":
            return [(kc1.name, kc2.name) for kc1,kc2 in settable_edges], [(kc1.name, kc2.name) for kc1,kc2 in non_settable_edges]
        elif edge_type == "id":
            return [(kc1.id, kc2.id) for kc1,kc2 in settable_edges], [(kc1.name, kc2.name) for kc1,kc2 in non_settable_edges]
        else:
            raise Exception(f'Edge type {edge_type} not supported.')
    
    def generate_probabilistic_requirements_map(self, global_proba_distrib:ProbaDistrib) -> "ProbabilisticRequirementsMap":
        probabilistic_requirements_dict:Dict[int, List[ProbabilisticRequirement]] = {}
        for kc_id, abstract_req_list in self.map.items():
            probabilistic_requirements_dict[kc_id] = []
            for abstract_req in abstract_req_list:
                if global_proba_distrib is None:
                    proba_distrib = None
                else:
                    mean = global_proba_distrib.sample(num=1)
                    proba_distrib = BinomialDistrib(mean=mean)
                probabilistic_requirement = abstract_req.generate_probabilistic_requirement(proba_distrib=proba_distrib)
                probabilistic_requirements_dict[kc_id].append(probabilistic_requirement)
        return ProbabilisticRequirementsMap(kc_id2obj=self.kc_id2obj, map=probabilistic_requirements_dict)


class ProbabilisticRequirementsMap(KCMap):
    def __init__(self, kc_id2obj:Dict[int,KC], map:Dict[int, List[ProbabilisticRequirement]]):
        super().__init__(kc_id2obj, map)
        self.map: Dict[int, List[ProbabilisticRequirement]]
    
    def generate_requirements_map(self, doc_id2obj:Dict[int,Doc]) -> "RequirementsMap":
        requirements_dict = {}
        for kc_id, proba_req_list in self.map.items():
            requirements_dict[kc_id] = [ proba_req.kc for proba_req in proba_req_list if proba_req.sample() ]

        return RequirementsMap(kc_id2obj=self.kc_id2obj, map=requirements_dict, doc_id2obj=doc_id2obj)


class RequirementsMap(KCMap):
    def __init__(self, kc_id2obj:Dict[int, KC], map:Dict[int, List[KC]], doc_id2obj:Dict[int,Doc]):
        super().__init__(kc_id2obj, map)
        self.doc_id2obj = doc_id2obj
        self.map: Dict[int, List[KC]]
    
    def to_kc_edges(self, edge_type="name") -> List[tuple]:
        kc2kc_edges = [ (kc, self.kc_id2obj[kc_id]) for kc_id,kc_list in self.map.items() for kc in kc_list]
        if edge_type == "name":
            return [ (kc1.name, kc2.name) for kc1,kc2 in kc2kc_edges]
        elif edge_type == "id":
            return [ (kc1.id, kc2.id) for kc1,kc2 in kc2kc_edges]
        else:
            raise Exception(f'Edge type {edge_type} not supported.')

    def get_requirements_of_kc(self, kc:KC) -> List[KC]:
        return self.map[kc.id]
    
    def to_doc_edges(self, edge_type="name"):
        raise NotImplementedError()
    
    def get_requirements_of_doc(self, doc:Doc) -> List[KC]:
        doc_requirements = []
        for kc in doc.kc_list:
            kc_requirements = self.get_requirements_of_kc(kc)
            for required_kc in kc_requirements:
                if (not required_kc in doc_requirements) and (not required_kc in doc.kc_list): # we only keep dependencies coming from outside the document
                    doc_requirements.append(required_kc)
        return doc_requirements
    
    def __str__(self):
        text_descriptor = self.text_descriptor()
        return text_descriptor
    def __repr__(self):
        text_descriptor = self.text_descriptor()
        return text_descriptor
    def text_descriptor(self):
        desription_dict = { self.kc_id2obj[kc_int].name:[kc.name for kc in kc_list] for kc_int,kc_list in self.map.items()}
        text = pprint.pformat(desription_dict)
        return text


class Knowledge:
    def __init__(self, kc_list:List[KC]):
        self.kc_list = kc_list
        self.kc_ids = [kc.id for kc in self.kc_list]
    
    def to_nodes(self, node_type="name") -> List[str]:
        if node_type == "name":
            return [kc.name for kc in self.kc_list]
        elif node_type == "id":
            return [kc.id for kc in self.kc_list]
        else:
            raise Exception(f'Node type "{node_type}" not supported.')
    
    def add_document(self, doc:Doc):
        for kc in doc.kc_list:
            self.maybe_add_kc(kc)

    def maybe_add_kc(self, kc:KC):
        if kc not in self.kc_list:
            self.kc_list.append(kc)
            self.kc_ids.append(kc.id)
    
    def duplicate(self) -> "Knowledge":
        kc_list = [kc for kc in self.kc_list]
        return Knowledge(kc_list=kc_list)
    
    def __contains__(self, item:KC):
        return item.id in self.kc_ids
    
    def __str__(self):
        text_descriptor = self.text_descriptor()
        return text_descriptor
    def __repr__(self):
        text_descriptor = self.text_descriptor()
        return text_descriptor
    def text_descriptor(self):
        text = str([kc.name for kc in self.kc_list])
        return text
    

class Feedback:

    DO_NOT_UNDERSTAND = 0
    UNDERSTAND        = 1
    TOO_EASY          = 2
    NOT_VISITED       = 3
    FEEDBACKS = [ DO_NOT_UNDERSTAND, UNDERSTAND, TOO_EASY, NOT_VISITED ]
    ID2MEANING = {NOT_VISITED:"not visited", DO_NOT_UNDERSTAND:"don't understand", UNDERSTAND:"understand", TOO_EASY:"too easy"}
    
    def __init__(self, id:int, learned_kc:List[KC]):
        assert id in Feedback.ID2MEANING.keys()
        self.id = id
        self.learned_kc = learned_kc
        self._tensor = None

        self.has_learned = self.id==Feedback.UNDERSTAND
    
    def interpret(self):
        return Feedback.ID2MEANING[self.id]
    
    @staticmethod
    def get_vector_size_and_range() -> Tuple[int, Tuple[int,int]]:
        return len(Feedback.FEEDBACKS), (0,1)
    
    def get_tensor(self, feedback_mode:str='default') -> torch.Tensor:
        if self._tensor is None:
            self._tensor = self.build_tensor(feedback_mode=feedback_mode)
        return self.tensor
    
    def get_learning_score(self):
        if len(self.learned_kc)==0:
            return 0
        return sum([kc.value for kc in self.learned_kc])

    @property
    def tensor(self):
        return self._tensor

    def build_tensor(self, feedback_mode:str) -> torch.Tensor:
        if feedback_mode=="default":
            return self.id2tensor(self.id)
        else:
            raise Exception(f'Feedback mode "{feedback_mode}" not supported.')
    
    @staticmethod
    def id2tensor(id:int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(id), len(Feedback.FEEDBACKS))



class Interaction:
    COLUMN_NAMES = ["doc_id", "feedback_id", "time"]
    ONLY_LAST_TIME_MODE = "only_last"
    RECORD_ALL_TIME_MODE = "record_all"
    TIME_MODES = [ONLY_LAST_TIME_MODE, RECORD_ALL_TIME_MODE]
    
    def __init__(self, doc:Doc, time_step:int, feedback:Feedback, feedback_mode:str):
        self.doc       = doc
        self.time_step = time_step
        self.feedback  = feedback
        self._feedback_tensor = self.feedback.get_tensor(feedback_mode=feedback_mode)
        
        self.df_data_dict = {"doc_id":self.doc.id, "feedback_id":self.feedback.id, "time":self.time_step}
    
    def get_tensor(self, time_mode:str, current_timestep:int=None) -> torch.Tensor:
        if time_mode=="none":
            self._tensor = self._feedback_tensor
        else:
            self._update_time_tensor(time_mode=time_mode, current_timestep=current_timestep)
            self._tensor = torch.cat(tensors=(self._feedback_tensor,self._time_tensor), dim=0)
        return self.tensor
    
    def get_label(self):
        return self.feedback.id
    
    def _update_time_tensor(self, time_mode:str, current_timestep:int):
        if time_mode==self.RECORD_ALL_TIME_MODE:
            self._time_tensor = torch.tensor([current_timestep-self.time_step])
        elif time_mode==self.ONLY_LAST_TIME_MODE:
            assert current_timestep is not None
            if current_timestep == self.time_step:
                self._time_tensor = torch.tensor([1])
            else:
                self._time_tensor = torch.tensor([0])
        else:
            raise Exception(f'Time mode "{time_mode}" not supported.')
    
    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor
    
    def has_learned(self, return_as_binary=False) -> bool:
        if return_as_binary:
            if self.feedback.has_learned:
                return 1
            else:
                return 0
        else:
            return self.feedback.has_learned
    
    @staticmethod
    def get_time_vector_size_and_range(time_mode:str) -> Tuple[int, Tuple[int,int]]:
        if time_mode=="none":
            return 0, None
        elif time_mode=="only_last":
            return 1, (0,1)
        elif time_mode=="record_all":
            return 1, (0,100)
        else:
            raise Exception(f'Time mode "{time_mode}" not supported.')
    
    @staticmethod
    def get_vector_size_and_range(time_mode:str) -> Tuple[int, Tuple[int,int]]:
        time_vector_size, time_vector_range = Interaction.get_time_vector_size_and_range(time_mode=time_mode)
        feedback_vector_size, feedback_vector_range = Feedback.get_vector_size_and_range()
        vector_size = time_vector_size+feedback_vector_size
        vector_range = (min(time_vector_range[0], feedback_vector_range[0]), max(time_vector_range[1], feedback_vector_range[1]))
        return vector_size, vector_range
    
    def __repr__(self) -> str:
        return self.text()
    def __str__(self) -> str:
        return self.text()
    def text(self) -> str:
        return str({"doc":self.doc.id, "feedback":self.feedback.interpret(), "timestep":self.time_step})


class NoneInteraction(Interaction):
    
    def __init__(self, doc:Doc, feedback_mode:str):
        feedback = Feedback(id=Feedback.NOT_VISITED, learned_kc=[])
        time_step = 0
        super().__init__(doc=doc, time_step=time_step, feedback=feedback, feedback_mode=feedback_mode)
        self._time_tensor = torch.tensor([self.time_step])
        self._tensor = torch.cat(tensors=(self._feedback_tensor,self._time_tensor), dim=0)
    
    def get_tensor(self, *args, **kwargs) -> torch.Tensor:
        return self.tensor


class InteractionsHistory:
    def __init__(self, docs_list:List[Doc], feedback_mode:str, doc2last_interaction:Dict[int,Interaction]=None):
        # self.dataframe = pd.DataFrame(columns=Interaction.COLUMN_NAMES)
        self.docs_list = docs_list
        self.feedback_mode = feedback_mode
        self.doc2last_interaction = {doc.id:NoneInteraction(doc=doc, feedback_mode=self.feedback_mode) for doc in self.docs_list} if (doc2last_interaction is None) else doc2last_interaction

        self.doc_ids_already_interacted = {doc.id for doc in self.docs_list if not isinstance(self.doc2last_interaction[doc.id], NoneInteraction)}
    
    def add(self, interaction:Interaction):
        # new_dict = interaction.df_data_dict
        # new_row = [ new_dict[col_name] for col_name in Interaction.COLUMN_NAMES ]
        # self.dataframe.iloc[-1] = new_row
        self.doc2last_interaction[interaction.doc.id] = interaction
        
        if not interaction.doc.id in self.doc_ids_already_interacted:
            self.doc_ids_already_interacted.add(interaction.doc.id)
    
    def duplicate(self) -> "InteractionsHistory":
        doc2last_interaction = {doc.id:self.doc2last_interaction[doc.id] for doc in self.docs_list}
        return InteractionsHistory(docs_list=self.docs_list, feedback_mode=self.feedback_mode, doc2last_interaction=doc2last_interaction)
    
    # def get_prev_interaction_with_doc(self, doc:Doc) -> Interaction:
    #     previous_interacts_with_doc = self.dataframe[self.dataframe['doc'] == doc.id]
    #     if not previous_interacts_with_doc.empty:
    #         interactions_data_dict = previous_interacts_with_doc.iloc[-1].to_dict()
    #         return Interaction(**interactions_data_dict)
    #     else:
    #         return None
    
    def to_vectors_dict(self, time_mode:str, current_timestep:int) -> Dict[int,torch.Tensor]:
        return {doc_id:interaction.get_tensor(time_mode=time_mode, current_timestep=current_timestep)
                for doc_id,interaction in self.doc2last_interaction.items()}

    def __repr__(self) -> str:
        return self.text()
    def __str__(self) -> str:
        return self.text()
    def text(self) -> str:
        return pprint.pformat(self.doc2last_interaction)