import experiments_utils as exputils

from . import rl_train
from .path_manager import PathManager

if __name__=="__main__":

    verbose  = 4
    
    quick_test = False
    use_wandb  = True
    use_sweep  = True
    
    exp_cfg = dict(
        exp_id        = 'rl_finetuning2',
        training_mode = 'non_linear_finetuning',                                                           # 'non_linear_finetuning' or 'pretraining'
        model_type    = 'GNNAgent',
        seed          = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
        device        = 'default',
        verbose       = verbose,
        version       = 'sept2024'
    )
    data_cfg = dict(
        kw_normalization         = ['none'],        # 'z-score' or 'unit' or 'none'
        exp_split_mode           = 'none', # 'none' or ...
        prior_knowledge_distrib  = ['zero', 'decreasing_exponential', 'uniform'],
        prereq_distrib           = 'uniform',
        prior_background_distrib = 'binomial', # 'binomial' or 'none' for linear corpora
        horizon_coef             = 1.,
        num_envs                 = 1,
        save_model               = False,
        save_best_fn             = False,
        non_linear_eval_env      = False
    )
    gnn_cfg = dict(
        # load_model    = ['pretrain/rl/pretrain_helpful-sweep-5', 'pretrain/rl/pretrain_quiet-sweep-3', 'none', 'pretrain/sl/pretrain_full1'], # 'none' or model name
        # load_model    = ['pretrain/expert/pretrain_full3', 'pretrain/sl/pretrain_full1', 'pretrain/rl/pretrain_quiet-sweep-3', 'none'],
        load_model    = 'pretrain/rl/pretrain_dry-sweep-2', # replace with 'none' for vassoyan et al.
        freeze_layers = "none",
        keep_sl_head  = [False],
        gnn_arch      = "transformer16",
        feedback_arch = "linear2",
        gnn_act       = "elu",
        feedback_act  = "relu",
        hidden_dim    = [128],
        heads         = 4,
        concat        = False,
        beta          = True,
        aggr          = 'add'
    )
    exp_cfg_trainer = dict(
        num_students        = 50, # num_students = max_epoch * step_per_epoch * episode_per_collect
        step_per_epoch      = 1,
        episode_per_collect = 5,   # 1
        episode_per_test    = 20,  # 10
        step_per_collect    = None
    )
    rl_cfg = dict(
        policy_name = "PGPolicy"
    )
    optimizer_cfg = dict(
        lr = [0.0005]
    )
    trainer_cfg = dict(
        repeat_per_collect = 15,
        batch_size         = 16
    )
    PGPolicy_cfg = dict(
        discount_factor      = [0.],
        ent_coef             = [0.],
        reward_normalization = [False],
        use_scheduler        = True,
        scheduler_factor     = 0.5,
        scheduler_patience   = 10,
        scheduler_threshold  = 0.5
    )
    replay_buffer_cfg = dict(
        multiply_factor =  [1.]
    )
    config = {**exp_cfg, **data_cfg, **gnn_cfg, **exp_cfg_trainer, **rl_cfg, **optimizer_cfg, **trainer_cfg, **PGPolicy_cfg, **replay_buffer_cfg}
    
    names_dict = {}
    wandb_names, metric_goal = rl_train.set_wandb_params(use_wandb=True, names_dict=names_dict)
    names_dict = {**names_dict, **wandb_names} 
    
    exputils.run_in_standard_mode(config=config, train_func=rl_train.train_func,
                                  quick_test=quick_test, use_sweep=use_sweep, use_wandb=use_wandb, is_offline=False,
                                  SYNC_WANDB_PATH=PathManager.SYNC_WANDB, names_dict=names_dict, metric_goal=metric_goal,
                                  sweep_trainer=rl_train.sweep_trainer, preprocess_quick_test_func=rl_train.preprocess_quick_test,
                                  wandb_method="grid")