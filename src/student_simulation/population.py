from typing import List, Iterator, Union, Tuple, Dict, Any, Sequence

import random
import networkx as nx

from ..path_manager import PathManager

from .student import Student

from .corpus_graph import CorpusGraph
from .types import BinomialDistrib, ProbaDistrib, ZeroDistrib, UniformDistrib,UniformDiscreteDistrib, DecreasingExponential
from .types import Knowledge, ProbabilisticRequirement, ProbabilisticRequirementsMap, RequirementsMap

from .. import utils


class Population:

    def __init__(self,
                 prior_knowledge_distrib:Union[ProbaDistrib, str],
                 prereq_distrib:Union[ProbaDistrib, str],
                 prior_background_distrib:Union[ProbaDistrib, str]=None,
                 feedback_mode:str='default',
                 prior_knowledge_distrib_params:dict=None,
                 prereq_distrib_min_max:tuple=(0.2, 0.4)
                 ):
        self.prior_knowledge_distrib     = self.maybe_generate_distrib(proba_distrib=prior_knowledge_distrib,  generator=self.generate_prior_knowledge_distrib, params=prior_knowledge_distrib_params)
        self.prior_background_distrib    = self.maybe_generate_distrib(proba_distrib=prior_background_distrib, generator=self.generate_prior_knowledge_distrib, params=None)
        self.prereq_distrib              = self.maybe_generate_distrib(proba_distrib=prereq_distrib,           generator=self.generate_prereq_distrib,          params=prereq_distrib_min_max)
        self.feedback_mode               = feedback_mode
        
        self.probabilistic_requirements_maps:Dict[str,ProbabilisticRequirementsMap] = {}
    

    def maybe_generate_distrib(self, proba_distrib:Union[ProbaDistrib, str], generator, params:dict=None):
        if isinstance(proba_distrib, str):
            return generator(proba_distrib, params)
        elif isinstance(proba_distrib, ProbaDistrib):
            return proba_distrib
        elif proba_distrib is None:
            return None
        else:
            raise Exception(f'Type "{type(proba_distrib)}" not supported for proba distribution.')
    

    def generate_prior_knowledge_distrib(self, prior_knowledge_distrib:str, params:dict=None):
        if prior_knowledge_distrib == "all":
            return None
        elif prior_knowledge_distrib == "none":
            return None
        elif prior_knowledge_distrib == "zero":
            return ZeroDistrib()
        elif prior_knowledge_distrib == "binomial":
            mean = params["mean"] if params is not None else 0.5
            return BinomialDistrib(mean=mean)
        elif prior_knowledge_distrib == "uniform":
            return UniformDiscreteDistrib()
        elif prior_knowledge_distrib == "decreasing_exponential":
            lam = params["lam"] if params is not None else 0.3
            return DecreasingExponential(lam=lam)
        else:
            raise Exception(f"Prior knowledge distribution '{prior_knowledge_distrib}' not supported.")
    
    def generate_prereq_distrib(self, prereq_distrib:str, params:dict):
        if prereq_distrib == "all":
            return None
        elif prereq_distrib == "uniform":
            min_val = params[0]
            max_val = params[1]
            return UniformDistrib(min_val=min_val, max_val=max_val)
        else:
            raise Exception(f"Prereq distribution '{prereq_distrib}' not supported.")
    

    def sample_students_iter(self, corpus_graph:CorpusGraph, num:Union[int,str]=1, verbose=0) -> Iterator[Student]:
        for prior_knowledge, requirements_map in self.sample_student_features(corpus_graph, num=num):
            student =  Student(prior_knowledge=prior_knowledge,
                               requirements_map=requirements_map,
                               corpus_graph=corpus_graph,
                               feedback_mode=self.feedback_mode,
                               verbose=verbose)
            yield student

    
    def sample_students(self, corpus_graph:CorpusGraph, num:Union[int,str]=1, verbose=0) -> List[Student]:
        students_list = [ Student(prior_knowledge=prior_knowledge,
                                  requirements_map=requirements_map,
                                  corpus_graph=corpus_graph,
                                  feedback_mode=self.feedback_mode,
                                  verbose=verbose)
                                  for prior_knowledge, requirements_map in self.sample_student_features(corpus_graph, num=num)]
        return students_list
    
    
    def sample_student_features(self, corpus_graph:CorpusGraph, num:Union[int,str]=1) -> Iterator[Tuple[List,List]]:
        if num=="all":
            assert corpus_graph.corpus_type==PathManager.LINEAR_CORPUS_TYPE, "Cannot generate all interactions in a non-linear corpus."

        corpus_id = corpus_graph.corpus_id  
        if corpus_graph.corpus_id not in self.probabilistic_requirements_maps.keys():
            self.probabilistic_requirements_maps[corpus_id] = corpus_graph.abstract_requirements_map.generate_probabilistic_requirements_map(global_proba_distrib=self.prereq_distrib)
        
        num_iter=1 if num=="all" else num
        for _ in range(num_iter):
            requirements_map = self.probabilistic_requirements_maps[corpus_id].generate_requirements_map(doc_id2obj=corpus_graph.doc_id2obj)
            num_prior_knowledge = num if num=="all" else 1
            # prior_knowledge_list = self.sample_prior_knowledge(requirements_map, corpus_graph, num=num_prior_knowledge)
            # for prior_knowledge in prior_knowledge_list:
            #         yield prior_knowledge, requirements_map
            for prior_knowledge in self.sample_prior_knowledge(requirements_map, corpus_graph, num=num_prior_knowledge):
                yield prior_knowledge, requirements_map


    def sample_prior_knowledge(self, requirements_map:RequirementsMap, corpus_graph:CorpusGraph, num:Union[int,str]=1):        
        regular_kc_names    = [kc.name for kc in corpus_graph.get_regular_kc()]
        background_kc_names = [kc.name for kc in corpus_graph.get_background_kc()]

        if num=='all':
            #TODO
            kc_names_list = [kc.name for kc in corpus_graph.kc_list]
            sorted_numbers = sorted([int(num) for num in kc_names_list])
            sorted_number_strings = [str(num) for num in sorted_numbers]
            sorted_kc = [ corpus_graph.kc_name2obj[kc_name] for kc_name in sorted_number_strings]
            for idx in range(len(sorted_number_strings)):
                kc_list = sorted_kc[:idx]
                prior_knowledge = Knowledge(kc_list=kc_list)
                yield prior_knowledge
        else:
            if len(background_kc_names)==0:
                sampled_background_kc_names_list = [[]]
            else:
                assert self.prior_background_distrib is not None, f'Background KC found in the corpus but you did not specify any prior background distribution.'
                num_samples_list = self.prior_background_distrib.sample(discrete_space_size=len(background_kc_names), num=num)
                num_samples_list = num_samples_list if isinstance(num_samples_list,list) else [num_samples_list]
                sampled_background_kc_names_list = [random.sample(background_kc_names, num_samples) for num_samples in num_samples_list]
        
            for sampled_background_kc_names in sampled_background_kc_names_list:
                not_sampled_background_kc_names = [kc_name for kc_name in background_kc_names if not (kc_name in sampled_background_kc_names) ]
                nodes = [ kc.name for kc in corpus_graph.kc_list]
                edges = requirements_map.to_kc_edges(edge_type="name")
                sorted_nodes = self.sort_nodes_according_to_prereq(nodes=nodes, edges=edges)
                eligible_kc_names_sorted = sorted_nodes.copy()
                for idx, name in enumerate(eligible_kc_names_sorted):
                    if name in not_sampled_background_kc_names:
                        eligible_kc_names_sorted = eligible_kc_names_sorted[:idx]
                        break
                eligible_regular_kc_names_sorted = [kc_name for kc_name in eligible_kc_names_sorted if kc_name in regular_kc_names ]
                if num=="all":
                    assert len(background_kc_names)==0
                    num_kc_known_list = list(range(len(eligible_regular_kc_names_sorted)))
                else:
                    num_kc_known_list = self.prior_knowledge_distrib.sample(discrete_space_size=len(eligible_regular_kc_names_sorted), num=num)

                if not isinstance(num_kc_known_list, list): num_kc_known_list = [num_kc_known_list]
                for num_kc_known in num_kc_known_list:
                    selected_nodes = eligible_regular_kc_names_sorted[:num_kc_known] + sampled_background_kc_names
                    kc_list = [ corpus_graph.kc_name2obj[node_name] for node_name in selected_nodes ]
                    prior_knowledge = Knowledge(kc_list=kc_list)

                    yield prior_knowledge
        

    def sort_nodes_according_to_prereq(self, nodes, edges):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The graph must be acyclic.")
        sorted_nodes = list(nx.topological_sort(G))
        return sorted_nodes

    # def select_kc_with_prerequisites(self, nodes:List[str], edges:List[Tuple[str,str]], num_nodes_to_select:int) -> List[str]:
    #     G = nx.DiGraph()
    #     G.add_nodes_from(nodes)
    #     G.add_edges_from(edges)
    #     if not nx.is_directed_acyclic_graph(G):
    #         raise ValueError("The graph must be acyclic.")
    #     sorted_nodes = list(nx.topological_sort(G))
    #     selected_nodes = sorted_nodes[:num_nodes_to_select]
    #     return selected_nodes
    

    def generate_all_action_reaction_pairs(self, corpus_graph_list:List[CorpusGraph]) -> Dict[str,Any]:
        corpus_id2action_reaction_pairs = {}
        for corpus_graph in corpus_graph_list:
            corpus_id = corpus_graph.corpus_id
            corpus_id2action_reaction_pairs[corpus_id] = self.generate_action_reaction_pairs_for_corpus(corpus_graph=corpus_graph, num="all")
        return corpus_id2action_reaction_pairs
    

    def generate_action_reaction_pairs_for_corpus(self, corpus_graph:CorpusGraph, num="all") -> List:
        action_reaction_pairs = []
        students_list = self.sample_students(corpus_graph=corpus_graph, num=num)
        for student in students_list:
            action_reaction_pairs = student.generate_all_action_reaction_pairs()
            action_reaction_pairs += action_reaction_pairs
        return action_reaction_pairs


if __name__ == "__main__":

    num_students = 30
    seed = 2
    
    # prior_knowledge_distrib  = DecreasingExponential(lam=0.3)
    prior_knowledge_distrib  = 'uniform'
    prereq_distrib = "uniform"
    prior_background_distrib = BinomialDistrib(mean=0.5)
    population = Population(prior_knowledge_distrib=prior_knowledge_distrib,
                            prior_background_distrib=prior_background_distrib,
                            prereq_distrib=prereq_distrib,
                            feedback_mode="default")
    ### corpus
    corpus_type = "non_linear"
    corpus_name="intro_to_ml"
    corpus_graph = CorpusGraph(corpus_name=corpus_name, corpus_group="hand_made", corpus_type=corpus_type)
    
    utils.set_all_seeds(seed=seed)
    students = population.sample_students(corpus_graph=corpus_graph, num=num_students)
    for student in students:
        print(student)
        student.display_knowledge_and_learning_prefs(include_documents=True)
    
    