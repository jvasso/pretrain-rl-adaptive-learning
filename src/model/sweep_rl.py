import wandb

from . import rl_train_old
from .. import utils


def sweep_rl_trainer(config_dict=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verbose = 4
    with wandb.init(config=config_dict) as run:
        rl_train_old.train_with_rl(config=wandb.config, num_envs=None, device=None, verbose=verbose, use_wandb=True, run=run)

def create_rl_sweep(parameters:dict, method:str):
    sweep_config = dict(method=method, metric={'name':'test/rew_mean','goal':'maximize'})
    sweep_config['parameters']=parameters
    sweep_id = wandb.sweep(sweep_config, entity='rl4edu', project="pre-trained-reco-system-project")
    return sweep_id


if __name__=="__main__":
    
    count = None
    method = "grid"
    
    exp_cfg = dict(
        exp_id                   = 'rl_finetuning1',
        training_mode            = 'non_linear_finetuning', # 'non_linear_finetuning' or 'pretraining'
        kw_normalization         = ['none'],        # 'z-score' or 'unit' or 'none'
        exp_split_mode           = 'none', # 'none' or ...
        prior_knowledge_distrib  = ['zero'],     # also try decreasing_exponential --> show that the model learns it!! (plot initial recommendations)
        prereq_distrib           = 'uniform',
        prior_background_distrib = 'binomial',        # 'binomial' or 'none' for linear corpora
        seed                     = [0,1,2,4,5],
        device                   = 'cuda',
        num_envs                 = 1
    )
    gnn_cfg = dict(
        load_model       = ['pretrain/sl/pretrain_full1'], # 'none' or model name
        keep_sl_head     = [False, True],
        freeze_layers    = ['none', 'all', 'all_except_last_gnn_module'], # 'none', 'all' or 'all_except_last_gnn_module'
        gnn_arch         = "transformer16",
        feedback_arch    = "linear2",
        gnn_act          = "elu",
        feedback_act     = "relu",
        hidden_dim       = [32],
        heads            = 2,
        concat           = False,
        beta             = True,
        aggr             = 'add'
    )
    exp_cfg_trainer = dict(
        num_students        = 100, # num_students = max_epoch * step_per_epoch * episode_per_collect
        step_per_epoch      = 1,
        episode_per_collect = 5,   # 1
        episode_per_test    = 20,  # 10
        step_per_collect    = None
    )
    rl_cfg = dict(
        policy_name = "PGPolicy"
    )
    optimizer_cfg = dict(
        lr = [0.0005, 0.001, 0.01]
    )
    trainer_cfg = dict(
        repeat_per_collect = 15,
        batch_size         = 16
    )
    PGPolicy_cfg = dict(
        discount_factor     = [0, 0.7],
        use_scheduler       = True,
        scheduler_factor    = 0.5,
        scheduler_patience  = 10,
        scheduler_threshold = 0.5
    )
    replay_buffer_cfg = dict(
        replay_buffer_size = 20000
    )
    parameters = {**exp_cfg, **exp_cfg_trainer, **gnn_cfg, **rl_cfg, **optimizer_cfg, **trainer_cfg, **PGPolicy_cfg, **replay_buffer_cfg}
    parameters = utils.to_sweep_format(parameters)

    sweep_id = create_rl_sweep(parameters=parameters, method=method)
    wandb.agent(sweep_id, sweep_rl_trainer, count=count)

   
