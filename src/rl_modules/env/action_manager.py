from typing import List, TYPE_CHECKING
import pprint
import time

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from .adaptive_learning_env import AdaptiveLearningEnv

from ...student_simulation import Student
from ...student_simulation.types import Feedback


class ActionManager:

    def __init__(self,
                 env:"AdaptiveLearningEnv",
                 action_mode:str="discrete",
                 action_params:dict=None):
        self.env               = env
        self.action_mode       = action_mode
        self.action_params     = action_params

        self._set_action_space()
    

    def set_current_action_space(self, num_docs:int):
        self.current_action_space = spaces.Discrete(n=num_docs)
    

    def _set_action_space(self):
        self.action_space = spaces.Discrete(n=self.env.MAX_NUM_DOCS)


    def do_action(self, action, time_step:int) -> Feedback:
        doc = self.interpret_action(action)
        interaction = self.env.student.interact_with_doc(doc, time_step=time_step)
        return interaction.feedback
    

    def interpret_action(self, raw_action):
        if self.action_mode == "human":
            doc_name = raw_action
            assert doc_name in self.env.corpus_graph.doc_name2obj.keys(), f'Document {doc_name} not in corpus {self.env.corpus_graph.corpus_id}'
            doc = self.env.corpus_graph.doc_name2obj[doc_name]
            return doc
        elif self.action_mode == "discrete":
            assert self.current_action_space.contains(raw_action)
            doc_id = raw_action
            doc = self.env.corpus_graph.doc_id2obj[doc_id]
            return doc
        else:
            raise Exception(f'Action mode "{self.action_mode}" not supported.')