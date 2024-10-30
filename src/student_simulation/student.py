from typing import Type, List, Dict, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

import torch
import networkx as nx

from .corpus_graph import CorpusGraph

from ..path_manager import PathManager
from . import utils_student_simul

from .types import KC, Doc, Interaction, Knowledge, RequirementsMap, InteractionsHistory, Feedback


class Student:

    def __init__(self,
                 prior_knowledge:Knowledge,
                 requirements_map:RequirementsMap,
                 corpus_graph:CorpusGraph,
                 feedback_mode:str="default",
                 knowledge:Knowledge=None,
                 interactions_history:InteractionsHistory=None,
                 verbose:int=4):
        
        self.prior_knowledge = prior_knowledge.duplicate()
        self.requirements_map = requirements_map
        self.corpus_graph = corpus_graph
        self.feedback_mode = feedback_mode
        self.verbose = verbose

        self.knowledge = prior_knowledge.duplicate() if (knowledge is None) else knowledge.duplicate()
        self.interactions_history = InteractionsHistory(docs_list=self.corpus_graph.doc_list, feedback_mode=self.feedback_mode) if (interactions_history is None) else interactions_history.duplicate()

        self._mastered_docs = [doc for doc in self.corpus_graph.doc_list if self.knows_doc(doc=doc)]
        self.accessible_docs, self.non_accessible_docs = self.search_accessible_docs()    

    # def init_knowledge_and_interactions(self):
    #     """ Set knowledge to prior knowledge and history of interactions to None """
    #     kc_list = [kc for kc in self.prior_knowledge.kc_list]
    #     self.knowledge = Knowledge(kc_list=kc_list)

    #     self.mastered_docs = [doc for doc in self.corpus_graph.doc_list if self.knows_doc(doc=doc)[0]]
    #     self.accessible_docs, self.non_accessible_docs = self.search_accessible_docs()

    #     self.interactions_history = InteractionsHistory(docs_list=self.corpus_graph.doc_list, feedback_mode=self.feedback_mode)

    
    def get_current_knowledge_docs(self) -> Dict[int,bool]:
        return {doc.id:self.knows_doc(doc=doc) for doc in self.corpus_graph.doc_list}


    def search_accessible_docs(self) -> Tuple[List[Doc],List[Doc]]:
        accessible_docs = []
        non_accessible_docs = []
        non_accessible_kc_list = self.search_non_accessible_kc()
        for doc in self.corpus_graph.doc_list:
            found_non_accessible_kc = False
            for kc in doc.kc_list:
                if kc in non_accessible_kc_list:
                    non_accessible_docs.append(doc)
                    found_non_accessible_kc = True
                    break
            if not found_non_accessible_kc:
                if not self.knows_doc(doc=doc):
                    accessible_docs.append(doc)
        return accessible_docs, non_accessible_docs
    

    def search_non_accessible_kc(self) -> List[KC]:
        nx_graph = self.build_networkx_graph()
        non_accessible_kc_list = []
        for background_kc in self.corpus_graph.get_background_kc():
            if background_kc not in self.knowledge:
                descendants_names=nx.descendants(nx_graph, background_kc.name)
                descendants_kc = [self.corpus_graph.kc_name2obj[kc_name] for kc_name in descendants_names]
                non_accessible_kc_list += descendants_kc
        return non_accessible_kc_list
        

    def has_requirements(self, doc:Doc):
        requirements = self.requirements_map.get_requirements_of_doc(doc=doc)
        for required_kc in requirements:
            if not self.knows_kc(kc=required_kc):
                return False
        return True
    

    def doc_id2feedback(self, feedback_mode:str, return_tensor=False) -> Dict[int,Union[Feedback,torch.Tensor]]:
        if return_tensor:
            return {doc_id:interaction.feedback.get_tensor(feedback_mode=feedback_mode) for doc_id,interaction in self.interactions_history.doc2last_interaction.items()}
        else:
            return {doc_id:interaction.feedback for doc_id,interaction in self.interactions_history.doc2last_interaction.items()}
    

    def has_interacted_with(self, doc_id:int):
        return doc_id in self.interactions_history.doc_ids_already_interacted

    def knows_kc(self, kc:KC) -> bool:
        return kc in self.knowledge

    def know_kc_name(self, kc_name:str):
        kc = self.corpus_graph.kc_name2obj[kc_name]
        return self.knows_kc(kc)
    
    def knows_doc(self, doc:Doc) -> bool:
        unknown_kc = []
        for kc in doc.kc_list:
            if not self.knows_kc(kc=kc):
                unknown_kc.append(kc)
        return len(unknown_kc)==0
    
    def get_unknown_kc_in_doc(self, doc:Doc) -> List[KC]:
        unknown_kc = []
        for kc in doc.kc_list:
            if not self.knows_kc(kc=kc):
                unknown_kc.append(kc)
        return unknown_kc
    

    def interact_with_doc(self,
                          doc:Doc,
                          time_step:int,
                          update_knowledge:bool=True,
                          update_interactions_hist:bool=True) -> Interaction:
        has_requirements = self.has_requirements(doc)
        unknown_kc_list = self.get_unknown_kc_in_doc(doc)
        already_has_knowledge = len(unknown_kc_list)==0
        if already_has_knowledge:
            feedback = Feedback(id=Feedback.TOO_EASY, learned_kc=[])
        else:
            if has_requirements:
                feedback = Feedback(id=Feedback.UNDERSTAND, learned_kc=unknown_kc_list)
                if update_knowledge:
                    self.learn_from_doc(doc=doc)
            else:
                feedback = Feedback(id=Feedback.DO_NOT_UNDERSTAND, learned_kc=[])
        interaction = Interaction(doc=doc,
                                  feedback=feedback,
                                  time_step=time_step,
                                  feedback_mode=self.feedback_mode)
        if update_interactions_hist:
            self.interactions_history.add(interaction=interaction)
        if self.verbose >= 4: print(f"Feedback: {interaction.feedback.interpret()}")
        return interaction
    
    def interact_with_doc_id(self, doc_id:int, time_step:int, update_knowledge:bool=True, update_interactions_hist:bool=True) -> Interaction:
        doc = self.corpus_graph.doc_id2obj[doc_id]
        return self.interact_with_doc(doc=doc, time_step=time_step, update_knowledge=update_knowledge, update_interactions_hist=update_interactions_hist)
    
    
    def learn_from_doc(self, doc:Doc):
        self.knowledge.add_document(doc)
        self._mastered_docs.append(doc)
        try:
            self.accessible_docs.remove(doc)
        except ValueError:
            raise ValueError(f"Document {doc} not found in the list")
    

    def has_completed_corpus(self):
        return len(self.accessible_docs)==0
    

    def generate_all_action_reaction_pairs(self):
        raise NotImplementedError()
    

    def build_networkx_graph(self):
        nodes = self.requirements_map.get_nodes()
        edges = self.requirements_map.to_kc_edges()
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)
        return nx_graph


    def display_knowledge_and_learning_prefs(self, include_documents:bool=True):
        nx_graph = self.build_networkx_graph()
        knowledge_nodes = self.knowledge.to_nodes()
        pos = utils_student_simul.grid_layout(nodes_list=nx_graph.nodes, two_dims=self.corpus_graph.corpus_type==PathManager.NON_LINEAR_CORPUS_TYPE)
        fig = plt.figure(figsize=(9,5))
        nx.draw(nx_graph, pos, with_labels=True)
        nx.draw_networkx_nodes(nx_graph, pos=pos, node_size=500, node_color="red")
        nx.draw_networkx_nodes(nx_graph, pos=pos, node_size=500, nodelist=knowledge_nodes, node_color="green")

        if include_documents:
            for doc_id, doc in self.corpus_graph.doc_id2obj.items():
                is_known = self.knows_doc(doc)
                doc_kc_nodes = [kc.name for kc in doc.kc_list]

                # Calculate the boundaries of the ellipse
                x_values, y_values = zip(*[pos[k] for k in doc_kc_nodes])
                center_x = sum(x_values) / len(doc_kc_nodes)
                center_y = sum(y_values) / len(doc_kc_nodes)
                width = max(x_values) - min(x_values)
                height = max(y_values) - min(y_values)

                height_padding, width_padding = self.set_padding()
                width_padding  = width + 2 * width_padding
                height_padding = height + 2 * height_padding
                xy = (center_x, center_y)
                ellipse_color = "green" if is_known else "red"
                ellipse = Ellipse(xy=xy, width=width_padding, height=height_padding, fill=True, edgecolor='black', facecolor=ellipse_color, alpha=0.2, linewidth=2)
                plt.gca().add_patch(ellipse)
        plt.show()
    

    def set_padding(self):
        if self.corpus_graph.corpus_type==PathManager.NON_LINEAR_CORPUS_TYPE:
            return 0.25, 0.25
        elif self.corpus_graph.corpus_type==PathManager.LINEAR_CORPUS_TYPE:
            return 0.0005, 0.4
        else:
            raise Exception(f'Corpus type "{self.corpus_graph.corpus_type}" not supported.')


    def get_feedback_features(self, time_mode:str, current_timestep:int, to_tensor:bool=False) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        feedback_features_dict = self.interactions_history.to_vectors_dict(time_mode=time_mode, current_timestep=current_timestep)
        if to_tensor:
            feedback_tensor = torch.stack([feedback_features_dict[doc.id] for doc in self.corpus_graph.doc_list], dim=0)
            return feedback_tensor
        else:
            return feedback_features_dict
    

    def look_ahead_next_interactions(self, timestep:int, to_labels=False, get_expert_policy:bool=False) -> Union[Dict[int, Interaction], torch.Tensor]:
        next_interactions_results = { doc.id:self.interact_with_doc(doc=doc,
                                                                    time_step=timestep,
                                                                    update_knowledge=False,
                                                                    update_interactions_hist=False)
                                                                    for doc in self.corpus_graph.doc_list}
        if to_labels:
            if get_expert_policy:
                return torch.tensor([next_interactions_results[doc.id].has_learned(return_as_binary=True) for doc in self.corpus_graph.doc_list])
            else:
                return torch.tensor([next_interactions_results[doc.id].get_label() for doc in self.corpus_graph.doc_list])
            # return torch.stack([next_interactions_results[doc.id].get_label() for doc in self.corpus_graph.doc_list], dim=0)
        return next_interactions_results
    

    # def get_past_feedback_tensor(self, time_mode:str) -> torch.Tensor:
    #     features_dict = self.interactions_history.to_vectors_dict(time_mode=time_mode, current_timestep=current_timestep)
    #     feedback_tensor = torch.tensor( for doc in self.corpus_graph.doc_list)
    #     return self.interactions_history.to_tensor()

    def duplicate(self) -> "Student":
        prior_knowledge = self.prior_knowledge.duplicate()
        requirements_map = self.requirements_map
        corpus_graph = self.corpus_graph
        feedback_mode = self.feedback_mode
        knowledge = self.knowledge.duplicate()
        interactions_history = self.interactions_history.duplicate()
        new_student = Student(prior_knowledge=prior_knowledge,
                              requirements_map=requirements_map,
                              corpus_graph=corpus_graph,
                              feedback_mode=feedback_mode,
                              knowledge=knowledge,
                              interactions_history=interactions_history,
                              verbose=0)
        return new_student


    def __str__(self):
        text_descriptor = self.text_descriptor()
        return text_descriptor
    def __repr__(self):
        text_descriptor = self.text_descriptor()
        return text_descriptor
    def text_descriptor(self):
        text = f"\nStudent:\n  • Knowledge: {self.knowledge}\n  • Learning preferences:\n {self.requirements_map}\n  • Interactions:\n {self.interactions_history}"
        return text




if __name__=="__main__":

    from .population import Population
    from .. import utils

    prior_knowledge_distrib     = "uniform"
    prereq_distrib              = "uniform"
    population = Population(prior_knowledge_distrib=prior_knowledge_distrib, prereq_distrib=prereq_distrib)
    
    corpus_type = "linear"
    corpus_group = "new_corpora"
    corpus_name="Artificial_Intelligence"
    corpus_graph = CorpusGraph(corpus_name=corpus_name, corpus_type=corpus_type, corpus_group=corpus_group)
    
    seed = 0
    utils.set_all_seeds(seed=seed)
    
    student = population.sample_students(corpus_graph=corpus_graph, num=1)[0]

    student.display_knowledge_and_learning_prefs()

    print(student)
    


    

