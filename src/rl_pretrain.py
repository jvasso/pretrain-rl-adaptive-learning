import os

from . import rl_train
from .path_manager import PathManager

from .custom_slurm_generator import CustomSlurmGenerator

import experiments_utils as exputils


if __name__=="__main__":

    verbose  = 4
    
    run_evaluation = True
    save_model     = False
    save_best_fn   = True

    quick_test = False
    use_wandb  = True
    use_sweep  = True
    
    exp_cfg = dict(
        exp_id        = 'rl_pretrain_longterm',
        training_mode = 'pretraining',     # 'non_linear_finetuning' or 'pretraining'
        model_type    = "GNNAgent",        # GNNAgent, Bassen
        seed          = [0,1,2,3],
        device        = 'default',
        verbose       = verbose,
        version       = 'sept2024'
    )
    data_cfg = dict(
        kw_normalization         = 'none',        # 'z-score' or 'unit' or 'none'
        exp_split_mode           = 'full_train2', # 'none' or ...
        prior_knowledge_distrib  = 'zero', # 'zero', 'uniform', 'decreasing_exponential'
        prereq_distrib           = 'uniform', 
        prior_background_distrib = 'binomial',    # 'binomial' or 'none' for linear corpora
        horizon_coef             = 1.,
        num_envs                 = 1,
        save_model               = save_model,
        save_best_fn             = save_best_fn,
        non_linear_eval_env      = True
    )
    gnn_cfg = dict(
        load_model    = 'pretrain/expert/pretrain_full2', # ['none', 'pretrain/sl/pretrain_full1', 'pretrain/rl/pretrain_full1', 'pretrain/sl/pretrain_expert_full2']
        freeze_layers = "none", # 'none', 'all' or 'all_except_last_gnn_module'
        keep_sl_head  = False,
        gnn_arch      = 'transformer16',
        feedback_arch = "linear2",
        gnn_act       = "elu",
        feedback_act  = "relu",
        hidden_dim    = 128,
        heads         = 4,
        concat        = False,
        beta          = True,
        aggr          = 'add'
    )
    exp_cfg_trainer = dict(
        num_students        = None, # 1500, num_students = max_epoch * step_per_epoch * episode_per_collect
        step_per_epoch      = 1,
        step_per_collect    = [1024], 
        max_epoch           = 30,
        episode_per_collect = None,
        episode_per_test    = 50   # 10
    )
    
    rl_cfg = dict(
        policy_name = "PGPolicy"
    )
    optimizer_cfg = dict(
        lr = [0.0001]
    )
    trainer_cfg = dict(
        repeat_per_collect = 15,
        batch_size         = 8
    )
    PGPolicy_cfg = dict(
        discount_factor      = [0.7],
        ent_coef             = [0.01],
        reward_normalization = [False],
        use_scheduler        = True,
        scheduler_factor     = 0.5,
        scheduler_patience   = 15,
        scheduler_threshold  = 0.5
    )
    replay_buffer_cfg = dict(
        replay_buffer_size = 20000
    )
    config = {**exp_cfg, **data_cfg, **exp_cfg_trainer, **gnn_cfg, **rl_cfg, **optimizer_cfg, **trainer_cfg, **PGPolicy_cfg, **replay_buffer_cfg}
    
    ### eval
    # if run_evaluation:
    #     parameters_sweep = utils.to_sweep_format(config)
    #     sweep_id = create_rl_sweep(parameters=parameters_sweep, method='grid')
    #     wandb.agent(sweep_id, rl_train.sweep_trainer, count=None)
    # else:
    #     sweep_id = None
    
    # if run_evaluation:

    arguments = exputils.retrieve_arguments()
    mode, names_dict, cluster_name = exputils.set_experiment_mode(arguments=arguments)
    wandb_names, metric_goal = rl_train.set_wandb_params(use_wandb=use_wandb, names_dict=names_dict)
    names_dict = {**names_dict, **wandb_names} if use_wandb else None
    
    if mode=="generate_slurm":
        filename = os.path.basename(__file__)
        filename = filename.split('.py')[0]
        exputils.generate_slurm(config=config, cluster_name=cluster_name, filename=filename, SlurmGenerator_cls=CustomSlurmGenerator)
    elif mode=="cluster":
        exputils.run_in_cluster_mode(train_func=rl_train.train_func, CONFIGS_PATH=PathManager.CONFIGS, SYNC_WANDB_PATH=PathManager.SYNC_WANDB, names_dict=names_dict)
    elif mode=="standard":
        exputils.run_in_standard_mode(config=config, train_func=rl_train.train_func,
                                    quick_test=quick_test, use_sweep=use_sweep, use_wandb=use_wandb, is_offline=False,
                                    SYNC_WANDB_PATH=PathManager.SYNC_WANDB, names_dict=names_dict, metric_goal=metric_goal,
                                    sweep_trainer=rl_train.sweep_trainer, preprocess_quick_test_func=rl_train.preprocess_quick_test,
                                    wandb_method="grid")
    else:
        raise ValueError(f'Mode {mode} not supported.')
    

    # if save_model:
    #     print('\n\n### Finished sweep: starting full pre-training\n')
        
    #     # ### full train
    #     new_parameters_dict = config.copy()
    #     new_parameters_dict['exp_split_mode'] = 'full_train2'
    #     new_parameters_dict['seed'] = 3
        
    #     config_object = SimpleNamespace(**new_parameters_dict)
    #     results_dict, model = rl_train.train_with_rl(config=config_object, num_envs=None, device=None, verbose=4, use_wandb=False)
    #     rl_train.save_rl_training(config_dict=new_parameters_dict, results_dict=results_dict, model=model, sweep_id=sweep_id)
