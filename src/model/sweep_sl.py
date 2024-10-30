import torch

import wandb

from . import sl_train_eval_old
from .. import utils


def sweep_sl_trainer(config_dict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verbose = 4
    with wandb.init(config=config_dict):
        sl_train_eval_old.train(config=wandb.config, device=device, verbose=verbose, use_wandb=True)

def create_sl_sweep(parameters:dict, method:str):
    sweep_config = dict(method=method, metric={'name':'eval/loss','goal':'minimize'})
    sweep_config['parameters']=parameters
    sweep_id = wandb.sweep(sweep_config, entity='rl4edu', project="pre-trained-reco-system-project")
    return sweep_id
  

if __name__=="__main__":

    count = None
    method = "grid"

    exp_cfg = dict(
         exp_dataset_name = 'first_dataset',
         exp_split_mode   = ['big1', 'big2'],
         seed             = [0]
    )
    gnn_cfg = dict(
        gnn_arch         = ['transformer16'],
        feedback_arch    = ['linear2'],
        gnn_act          = 'elu',
        feedback_act     = ['relu'],
        hidden_dim       = [128],
        heads            = [2, 4],
        f_dropout_mode   = ['all_except_last'],
        gnn_dropout_mode = ['last_only'],
        dropout_features = [0, 0.2],
        dropout_conv     = [0.5, 0.6],
        dropout_dense    = [0.2],
        dropout_f        = [0.3],
        edge_dropout     = [0, 0.1],
        concat           = [True],
        beta             = True,
        aggr             = ['add']
    )
    training_cfg = dict(
        batch_size_train = 4,
        batch_size_valid = 4,
        optimizer_name   = 'adam',
        lr               = [0.01, 0.001],
        num_epochs       = 5,
        weight_decay     = [5e-4, 5e-3]
    )
    data_cfg = dict(
        kw_normalization = ['none']
    )
    scheduler_cfg = dict(
        scheduler_factor=0.1,
        scheduler_patience=1, 
        scheduler_threshold=0.02
    )

    parameters = {**exp_cfg, **gnn_cfg, **training_cfg, **data_cfg, **scheduler_cfg}
    parameters['log_cf_mx_img'] = False
    parameters = utils.to_sweep_format(parameters)
    sweep_id   = create_sl_sweep(parameters=parameters, method=method)
    wandb.agent(sweep_id, sweep_sl_trainer, count=count)

   
