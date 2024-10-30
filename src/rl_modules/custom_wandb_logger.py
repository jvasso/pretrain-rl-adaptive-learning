import copy
import pprint

import torch

# from tianshou.utils.logger.wandb import WandbLogger
from tianshou.utils.logger.base import BaseLogger
import numpy as np

from typing import Callable, Dict, Optional, Tuple, Union

import wandb


class CustomWandbLogger(BaseLogger):

    TRAIN  = "train"
    TEST   = "test"
    STAGES = [TRAIN, TEST]

    STEP_COUNT = "step"
    EP_COUNT   = "episode"
    
    REW = "rew"
    LOSS_METRICS  = []
    SCORE_METRICS = [REW]
    
    def __init__(
        self,
        optimizer:torch.optim.Optimizer,
        train_interval_ep : int = 1,
        test_interval_ep  : int = 1,
        update_interval_ep: int = 1,
        use_wandb:bool=True
    ):
        super().__init__()
        
        self.optimizer          = optimizer
        self.train_interval_ep  = train_interval_ep
        self.test_interval_ep   = test_interval_ep
        self.update_interval_ep = update_interval_ep

        self.use_wandb = use_wandb
        
        self.last_log_train_ep  = 0
        self.last_log_test_ep   = 0
        self.last_log_update_ep = 0

        self.train_episode_count = 0
        self.train_step_count    = 0

        self.current_step_log = {self.EP_COUNT:0, self.STEP_COUNT:0}
        self.logs_history = []
    
    
    def write(self, step_type: str, step: int, new_log_data):
        """Specify how the writer is used to log data.
        
        :param str step_type: namespace which the data dict belongs to. --> "train/env_ep", "test/env_ep"
        :param int step: stands for the ordinate of the data dict.
        :param dict data: the data to write with format ``{key: value}``.
        """
        if step_type != "update/gradient_step":
            
            if step_type=='train':
                self.current_step_log = {self.EP_COUNT:self.train_episode_count, self.STEP_COUNT:self.train_step_count}
            self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, new_log_data)
            
            if not 'current_lr' in self.current_step_log:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, {'current_lr':current_lr})
            
            if self.use_wandb:
                wandb.log(copy.deepcopy(self.current_step_log))
        
        else:
            self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, new_log_data)
            
            if self.use_wandb:
                wandb.log(copy.deepcopy(self.current_step_log))

    
    def extract_relevant_log_data(self, collect_result: dict, env_type:str):
        log_data = {
                f"{env_type}/{self.REW}_mean": collect_result["rew"],
                f"{env_type}/{self.REW}_std": collect_result["rew_std"],
                f"{env_type}/length_mean": collect_result['len'],
                f"{env_type}/length_std": collect_result['len_std'],
                f"{env_type}/n_collected_episodes": collect_result["n/ep"],
                f"{env_type}/n_collected_steps": collect_result["n/st"],
        }
        return log_data
    
    
    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.
        
        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        assert collect_result["n/st"] > 0
        self.train_episode_count += collect_result["n/ep"]
        self.train_step_count    += collect_result["n/st"]
        log_data = self.extract_relevant_log_data(collect_result=collect_result, env_type=self.TRAIN)
        self.write(self.TRAIN, step, log_data)

    
    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        assert sum(collect_result['lens'])==collect_result['n/st']
        
        log_data = self.extract_relevant_log_data(collect_result=collect_result, env_type=self.TEST)
        self.write(self.TEST, step, log_data)
    

    @staticmethod
    def safe_update(original_dict, new_entries):
        for key in new_entries:
            if key in original_dict:
                raise KeyError(f"Key '{key}' already exists in the dictionary.")
            original_dict[key] = new_entries[key]
        return original_dict


    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_update_step >= self.update_interval:
            log_data = {f"update/{k}": v for k, v in update_result.items()}
            self.write("update/gradient_step", step, log_data)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        pass
    
    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        pass

