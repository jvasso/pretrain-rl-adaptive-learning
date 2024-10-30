from types import SimpleNamespace

import torch

from experiments_utils import run_in_standard_mode

from . import sl_train

if __name__=="__main__":

    run_evaluation = False
    verbose = 4

    exp_cfg = dict(
        exp_id    = 'expert_policy_full_pretrain2',
        seed      = [0],
        device    = 'default',
        init_eval = True,
        verbose   = verbose,
        version    = 'sept2024'
    )
    dataset_cfg = dict(
        exp_dataset_name = "expert_policy2", # first_dataset
        prediction_type  = "next_reco",
        exp_split_mode   = ["big1", "big2", "big3"],
    )
    gnn_cfg = dict(
        gnn_arch         = "transformer16",
        feedback_arch    = "linear2",
        gnn_act          = "elu",
        feedback_act     = "relu",
        hidden_dim       = 128,
        heads            = 4,
        f_dropout_mode   = 'all_except_last',
        gnn_dropout_mode = 'last_only',
        dropout_features = 0.2,
        dropout_conv     = 0.6,
        dropout_dense    = 0.2,
        dropout_f        = 0.4,
        edge_dropout     = 0.1,
        concat           = False,
        beta             = True,
        aggr             = 'add'
    )
    training_cfg = dict(
        batch_size_train = 4, # 128
        batch_size_valid = 8, # 256
        add_softmax      = False,
        optimizer_name   = 'adam',
        lr               = 0.001,
        num_epochs       = 2,
        weight_decay     = 0.005
    )
    data_cfg = dict(
        kw_normalization = "none"
    )
    scheduler_cfg = dict(
        scheduler_factor=0.1,
        scheduler_patience=1, 
        scheduler_threshold=0.02
    )
    config  = {**exp_cfg, **dataset_cfg, **gnn_cfg, **training_cfg, **data_cfg, **scheduler_cfg}
    # parameters_dict['log_cf_mx_img']=True
    
    ### eval
    if run_evaluation:
        names_dict = {}
        wandb_names, metric_goal = sl_train.set_wandb_params(use_wandb=True, names_dict=names_dict)
        names_dict = {**names_dict, **wandb_names} 
        run_in_standard_mode(config=config, train_func=sl_train.train_func,
                            quick_test=False, use_sweep=True, use_wandb=True, is_offline=False,
                            SYNC_WANDB_PATH=None, names_dict=names_dict, metric_goal=metric_goal,
                            sweep_trainer=sl_train.sweep_trainer, preprocess_quick_test_func=None,
                            wandb_method="grid")
    else:
        sweep_id = None

    print('\n\n### Finished sweep: starting full pre-training\n')
    
    ### full train
    new_config_dict = config.copy()
    new_config_dict['init_eval'] = False
    new_config_dict['exp_split_mode'] = 'full_train2'
    new_config_dict['seed'] = 0
    
    config_object = SimpleNamespace(**new_config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_loader, eval_loader, results_dict, train_cf_mx, eval_cf_mx = sl_train.train_func(config=config_object, use_wandb=False)
    sl_train.save_sl_training(config_dict=new_config_dict, results_dict=results_dict, model=model, sweep_id=sweep_id)