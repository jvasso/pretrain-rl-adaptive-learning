from typing import List, Union

import gymnasium as gym

import torch
import math

from .action_manager import ActionManager
from .reward_manager import RewardManager
from .observation_manager import ObservationManager

from ...student_simulation import CorpusGraph, Population, Student



class AdaptiveLearningEnv(gym.Env):

    metadata = {"render_modes": ["terminal", "graph"]}
    MAX_NUM_DOCS = 200 # for optimization purposes
    MAX_NUM_KW   = 200

    STEP_COUNT = 0
    
    def __init__(self,
                 corpus_graphs_list:List[CorpusGraph],
                 population:Union[str, Population]="default",
                 action_manager:Union[str, ActionManager]="default",
                 reward_manager:Union[str, RewardManager]="default",
                 obs_manager:Union[str, ObservationManager]="default",
                 time_mode:str="only_last",
                 feedback_mode:str="default",
                 horizon_mode:str="target_kc",
                 horizon_coef:float=1.,
                 target_kc_mode:str='default',
                 is_eval:bool=False,
                 human_mode:bool=False,
                 seed:int=None,
                 render_mode:str=None,
                 verbose:int=4):
        
        self.corpus_graphs_list = corpus_graphs_list

        self.time_mode      = time_mode
        self.feedback_mode  = feedback_mode
        self.horizon_mode   = horizon_mode
        self.horizon_coef = horizon_coef
        self.target_kc_mode = target_kc_mode

        self.is_eval    = is_eval
        self.human_mode = human_mode
        self.seed       = seed
        self.verbose    = verbose
        
        self._init_corpus() # to retrieve some metadata (features size etc.)
        self.population     = self.generate_population(population)         if isinstance(population,str)     else population
        self.action_manager = self.generate_action_manager(action_manager) if isinstance(action_manager,str) else action_manager
        self.reward_manager = self.generate_reward_manager(reward_manager) if isinstance(reward_manager,str) else reward_manager
        self.obs_manager    = self.generate_obs_manager(obs_manager)       if isinstance(obs_manager,str)    else obs_manager

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # action & observation spaces
        self.action_space      = self.action_manager.action_space
        self.observation_space = self.obs_manager.observation_space

        # statistics
        self.done_count = 0
    

    def reset(self, seed:int=0):
        if self.verbose >=5: print(f'\nReset {"eval" if self.is_eval else ""}')
        super().reset(seed=seed)
        self._init_corpus()
        self._init_student()
        self.action_manager.set_current_action_space(num_docs=self.num_docs)
        
        self.target_kc_list = self.set_target_kc()
        self.horizon        = self.set_horizon()
        self._init_episode_stats()

        infos = {}
        obs = self.obs_manager.get_obs()
        self.obs = obs
        # print(f'send obs of shape {obs.X_feedback.shape}')
        return obs, infos
    

    def step(self, action):
        feedback = self.action_manager.do_action(action, time_step=self.step_count)
        terminated, cause = self.is_terminated()
        reward, reward_infos = self.reward_manager.compute_reward(action=action, terminated=terminated, feedback=feedback)
        self.cumul_reward += reward
        
        obs = self.obs_manager.get_obs()
        infos = {"update_reward":reward_infos}
        
        if self.verbose >= 5: print(f'Step{" eval" if self.is_eval else ""}: {self.step_count:>2} | Action: {action:>2} | Reward: {round(reward, ndigits=2):>2} {reward_infos}')
        if self.verbose >= 5 and terminated: print(f"Terminated (cause: {cause})")
        
        self.step_count += 1
        AdaptiveLearningEnv.STEP_COUNT += 1

        return obs, reward, terminated, False, infos
    

    def is_terminated(self):
        if self.step_count == self.horizon - 1:
            return True, f"ep max length ({self.step_count})"
        elif self.knows_all_target_kc():
            return True, "knows all target kc"
        elif self.corpus_completed():
            return True, "corpus completed"
        else:
            return False, ""
    

    def corpus_completed(self):
        return self.student.has_completed_corpus()
    

    def knows_all_target_kc(self):
        for kc in self.target_kc_list:
            if not self.student.knows_kc(kc=kc):
                return False
        return True


    def generate_population(self, population:str) -> Population:
        if population == "default":
            prior_knowledge_distrib = "uniform"
            prereq_distrib = "uniform"
            population = Population(prior_knowledge_distrib=prior_knowledge_distrib,
                                    prereq_distrib=prereq_distrib,
                                    feedback_mode=self.feedback_mode)
            return population
        else:
            raise Exception(f'Population type "{population}" not supported.')


    def generate_action_manager(self, action_manager:str) -> ActionManager:
        if action_manager=="default":
            if self.human_mode:
                return ActionManager(env=self, action_mode="human")
            else:
                return ActionManager(env=self)
        else:
            raise Exception(f'Action manager "{action_manager}" not supported.')
    
    def generate_reward_manager(self, reward_manager:str) -> RewardManager:
        if reward_manager=="default":
            reward_coeffs = {"progress":1}
            return RewardManager(env=self, reward_coeffs=reward_coeffs)
        elif reward_manager=='eval':
            return RewardManager(env=self, reward_coeffs={'eval':1})
        else:
            raise Exception(f'Reward manager "{reward_manager}" not supported.')
    
    def generate_obs_manager(self, obs_manager:str) -> ObservationManager:
        if obs_manager=="default":
            return ObservationManager(env=self, type=obs_manager)
        elif obs_manager=="bassen":
            return ObservationManager(env=self, type=obs_manager)
        else:
            raise Exception(f'Observation manager "{obs_manager}" not supported.')
    

    def _init_corpus(self):
        self.corpus_idx = int(torch.randint(low=0, high=len(self.corpus_graphs_list), size=(1,)))
        self.corpus_graph = self.corpus_graphs_list[self.corpus_idx]
        # if self.verbose >= 1: print(f'Sampled corpus {self.corpus_graph.corpus_name} (num docs: {self.corpus_graph.num_docs})')
        assert isinstance(self.corpus_graph, CorpusGraph)
        self.num_docs = self.corpus_graph.num_docs
        self.num_kw   = self.corpus_graph.num_kw
    

    def _init_student(self):
        students_list = self.population.sample_students(corpus_graph=self.corpus_graph, num=1)
        self.student = students_list[0]
    

    def set_target_kc(self):
        if self.target_kc_mode=='default':
            target_kc_list = [kc for kc in self.corpus_graph.kc_list if kc.is_regular() and kc.level=='1']
            return target_kc_list
        else:
            raise ValueError(f'Target KC mode {self.target_kc_mode} not supported.')
            

    def set_horizon(self):
        if isinstance(self.horizon_mode, int):
            return self.horizon_mode
        elif self.horizon_mode=="num_accessible_docs":
            return int(math.floor(len(self.student.accessible_docs)*self.horizon_coef))
        elif self.horizon_mode=="target_kc":
            horizon = int(math.floor(len(self.target_kc_list)*self.horizon_coef))
            return horizon
        else:
            raise Exception(f'Horizon mode {self.horizon_mode} not supported.')


    def _init_episode_stats(self):
        self.cumul_reward = 0
        self.step_count = 0
        self.past_actions = []
        self.trajectory_length = 0


    def render(self):
        # Render the environment to the screen
        # do something with trajectory visualization
        if self.render_mode == "graph":
            return self.render_graph()
    
    def render_graph(self):
        self.student.display_knowledge_and_learning_prefs(include_documents=True)
    

    # @staticmethod
    # def generate_env(corpus_graph_list:List[CorpusGraph],
    #                  prior_knowledge_distrib:str="uniform", prereq_distrib:str="uniform", prior_background_distrib:str=None,
    #                  prereq_distrib_min_max:tuple=(0.2,0.4),
    #                  is_eval:bool=False, verbose=0):
    #     population = Population(prior_knowledge_distrib=prior_knowledge_distrib,
    #                             prereq_distrib=prereq_distrib,
    #                             prior_background_distrib=prior_background_distrib,
    #                             prereq_distrib_min_max=prereq_distrib_min_max)
    #     env = AdaptiveLearningEnv(corpus_graphs_list=corpus_graph_list,
    #                             population=population,
    #                             human_mode=False,
    #                             horizon_mode="target_kc",
    #                             render_mode=None,
    #                             is_eval=is_eval,
    #                             verbose)
    #     return env
    

if __name__=="__main__":
    from ... import utils
    from ...student_simulation.types import BinomialDistrib, ProbaDistrib, ZeroDistrib, UniformDistrib,UniformDiscreteDistrib, DecreasingExponential
    
    ### non linear setting
    corpus_types  = "non_linear"
    corpus_groups = "hand_made"
    corpus_names = "intro_to_ml"
    prior_background_distrib = BinomialDistrib(mean=0.5)
    
    ### linear setting
    # corpus_types  = "linear"
    # corpus_groups = "hand_made"
    # # corpus_names  = "corpus2"
    # corpus_names = 'all'
    # prior_background_distrib = None
    
    num_episodes = 1
    
    seed = 21
    verbose = 5
    utils.set_all_seeds(seed=seed)
    
    # prior_knowledge_distrib  = DecreasingExponential(lam=0.3)
    prior_knowledge_distrib  = 'uniform'
    prereq_distrib           = "uniform"
    
    
    population = Population(prior_knowledge_distrib=prior_knowledge_distrib,
                            prior_background_distrib=prior_background_distrib,
                            prereq_distrib=prereq_distrib,
                            feedback_mode="default")
    
    corpus_graph_list = CorpusGraph.generate_corpus_graph_list(corpus_types=corpus_types, corpus_groups=corpus_groups, corpus_names=corpus_names)
    env = AdaptiveLearningEnv(corpus_graphs_list=corpus_graph_list,
                              population   = population,
                              human_mode   = True,
                              horizon_mode = "target_kc",
                              render_mode  = "graph",
                              verbose      = verbose)
    for _ in range(num_episodes):
        obs, info = env.reset(seed=seed)
        print(f'Student knowledge: {env.student.knowledge}')
        terminated = False
        while not terminated:
            env.render()
            action = input("Please recommend doc:")
            obs, reward, terminated, truncated, info = env.step(action)

