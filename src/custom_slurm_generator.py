import os
from typing import List

import experiments_utils as exputils

from .path_manager import PathManager


class CustomSlurmGenerator(exputils.SlurmGenerator):

    # mandatory attributes
    PROJECT_PATH            = PathManager.PROJECT
    CONFIGS_PATH            = PathManager.CONFIGS
    SLURM_PATH              = PathManager.SLURM
    LOGFILES_PATH           = PathManager.LOGFILES
    SYNC_WANDB_PATH         = PathManager.SYNC_WANDB
    TRAIN_FILES_FOLDER_PATH = PathManager.SRC
    CONDA_ENV_NAME = 'rl4edu'
    EMAIL          = ''
    
    # mandatory attributes for RUCHE
    ANACONDA_MODULE_RUCHE = 'anaconda3/2024.06/gcc-13.2.0'
    CUDA_MODULE_RUCHE     = 'cuda/12.2.1/gcc-11.2.0'
    REPO_PATH_RUCHE       = '/gpfs/workdir/vassoyanj/repos/rl4edu'
    CONDA_ENV_PATH_RUCHE  = '~/.conda/envs/rl4edu'
    
    # mandatory attributes for JEAN-ZAY
    ANACONDA_MODULE_JEAN_ZAY = 'anaconda-py3/2023.09'
    
    @staticmethod
    def adjust_config_to_constraints(config:dict, slurm_kwargs:dict, cluster_name:str):
        
        if cluster_name == CustomSlurmGenerator.CLUSTER_JEAN_ZAY:
            constraint = slurm_kwargs['constraint']
            if '32g' in constraint:
                config["num_envs"]   = 1
                config["batch_size"] = 16
            elif 'a100' in constraint:
                config["num_envs"]   = 1
                config["batch_size"] = 32
            else:
                raise ValueError(f'Constraint {constraint} not supported.')
        
        elif cluster_name == CustomSlurmGenerator.CLUSTER_RUCHE:
            partition = slurm_kwargs['partition']
            if partition == 'gpu':
                config["num_envs"]   = 1
                config["batch_size"] = 8
            elif partition == 'gpua100':
                config["num_envs"]   = 1
                config["batch_size"] = 16
            else:
                raise ValueError(f'Partition {partition} not supported.')
        else:
            raise ValueError(f'Cluster name {cluster_name} not supported.')

        return config
    

    @staticmethod
    def shorten_cluster_name(cluster_name:str):
        if cluster_name == CustomSlurmGenerator.CLUSTER_JEAN_ZAY:
            return 'JZ'
        elif cluster_name == CustomSlurmGenerator.CLUSTER_RUCHE:
            return 'R'
        else:
            raise ValueError(f'Cluster name {cluster_name} not supported.')
        