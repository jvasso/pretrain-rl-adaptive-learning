from typing import List, TYPE_CHECKING

from ...student_simulation.student import Student
from ...student_simulation.types import Feedback

if TYPE_CHECKING:
    from .adaptive_learning_env import AdaptiveLearningEnv


class RewardManager:
    
    def __init__(self,
                 env:"AdaptiveLearningEnv",
                 reward_coeffs:dict={"progress":1, "time":0.1},
                 include_difficulty_level:bool=True,
                 ):
        self.reward_id2func = {"progress":self.progress_reward,
                               "time":self.elapsed_time_reward}
        assert set(reward_coeffs.keys()).issubset(self.reward_id2func.keys())
        
        self.env = env
        self.rewards_coeffs = self.preprocess_rewards_coeffs(reward_coeffs)
        self.include_difficulty_level = include_difficulty_level


    # def estimate_max_reward(self, student:Student):
    #     if self.include_difficulty_level:
    #         raise NotImplementedError()
    #     else:
    #         num_accessible_docs = len(student.accessible_docs)
    #         if self.env.horizon < num_accessible_docs:
    #             num_accessible_docs = self.env.horizon
    #         rewards = [ self.compute_reward(force_positive=True)[0] for accessed_doc in range(num_accessible_docs) ]
    #         return sum(rewards)


    def preprocess_rewards_coeffs(self, coeffs:dict):
        coeffs_sum = 0
        for reward_id, reward_coeff in coeffs.items():
            if reward_id not in self.reward_id2func:
                raise Exception(f"Reward id {reward_id} not supported.\nSupported ids: {list(self.reward_id2func.keys())}")
            coeffs_sum += reward_coeff
        # preprocessed_coeffs = {id:coeff/coeffs_sum for id, coeff in coeffs.items()}
        preprocessed_coeffs = {id:coeff for id, coeff in coeffs.items()}
        return preprocessed_coeffs
    
    
    def compute_reward(self, action=None, terminated:bool=None, feedback:Feedback=None, force_positive:bool=False):
        reward = 0
        reward_info = {}
        for reward_id, reward_coeff in self.rewards_coeffs.items():
            reward_func = self.reward_id2func[reward_id]
            reward_term, causes_dict = reward_func(action=action, terminated=terminated, feedback=feedback, force_positive=force_positive)
            reward += reward_coeff*reward_term
            reward_info[reward_id] = {"val":round(reward_term,2), **causes_dict}
        # reward_info = {"estimate_damage":self.estimate_damage()}
        return reward, reward_info
    
    ### REWARDS
    def elapsed_time_reward(self, **kwargs):
        return -1, {}

    def progress_reward(self, action, terminated:bool, feedback:Feedback, force_positive:bool=False):
        if force_positive:
            return 1, {}
        if self.include_difficulty_level:
            return feedback.get_learning_score(), {}
        else:
            if feedback.has_learned:
                return 1, {}
            return 0, {}