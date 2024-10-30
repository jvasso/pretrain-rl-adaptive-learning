from typing import List
import os

from math import nan

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader.dataloader import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import wandb

import experiments_utils as exputils

from .path_manager import PathManager
from .custom_slurm_generator import CustomSlurmGenerator

from .model import GNNAgent
from .student_simulation.corpus_graph_dataset import CorpusGraphDataset
from .student_simulation.types import Feedback
from .custom_slurm_generator import CustomSlurmGenerator

LOSS            = 'loss'
ACC             = 'acc'
PRECISION_MACRO = 'precision_macro'
RECALL_MACRO    = 'recall_macro'
F1_WEIGHTED     = 'f1_weighted'
F1_MACRO        = 'f1_macro'
# F1_MACRO_02     = 'f1_macro_02'
F1_0            = 'f1_cls0'
F1_1            = 'f1_cls1'
F1_2            = 'f1_cls2'

RIGHT_RECO_PROBA = 'right_reco_prob'

TRAIN = 'train'
EVAL  = 'eval'

LOSS_METRICS  = [LOSS]
SCORE_METRICS = [ACC, PRECISION_MACRO, RECALL_MACRO, F1_WEIGHTED, F1_MACRO, F1_0, F1_1, F1_2, RIGHT_RECO_PROBA]
STAGES = [TRAIN, EVAL]

CLASS_NAMES_FEEDBACK = [ Feedback.ID2MEANING[feedback_id] for feedback_id in Feedback.FEEDBACKS if feedback_id != Feedback.NOT_VISITED ]
CLASS_NAMES_RECO     = ["not reco", "reco"]

DEFAULT_WANDB_GROUP_NAME = 'sl_pretraining'



def train_func(config, use_wandb, run=None):
    device = exputils.preprocess_training(config=config, seed=config.seed, device=config.device)
    exputils.maybe_define_wandb_metrics(loss_metrics=LOSS_METRICS, score_metrics=SCORE_METRICS, stages=STAGES, use_wandb=use_wandb, custom_step_metric=None)
    
    train_graphs, eval_graphs = CorpusGraphDataset.generate_train_eval_corpus_graphs(dataset_name=config.exp_dataset_name, split_mode=config.exp_split_mode)
    train_dataset = CorpusGraphDataset(corpus_list=train_graphs, dataset_name=config.exp_dataset_name, kw_normalization=config.kw_normalization, is_eval=False)
    eval_dataset  = CorpusGraphDataset(corpus_list=eval_graphs,  dataset_name=config.exp_dataset_name, kw_normalization=config.kw_normalization, is_eval=True) if len(eval_graphs)>0 else None
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True)
    eval_loader   = DataLoader(eval_dataset, batch_size=config.batch_size_valid, shuffle=False) if len(eval_graphs)>0 else None
    data_sample = train_dataset[0]
    model = GNNAgent(gnn_arch         = config.gnn_arch,
                     feedback_arch    = config.feedback_arch,
                     gnn_act          = config.gnn_act,
                     feedback_act     = config.feedback_act,
                     hidden_dim       = config.hidden_dim,
                     heads            = config.heads,
                     f_dropout_mode   = config.f_dropout_mode,
                     gnn_dropout_mode = config.gnn_dropout_mode,
                     dropout_features = config.dropout_features,
                     dropout_conv     = config.dropout_conv,
                     dropout_dense    = config.dropout_dense,
                     dropout_f        = config.dropout_f,
                     aggr             = config.aggr,
                     edge_dropout     = config.edge_dropout,
                     concat           = config.concat,
                     beta             = config.beta,
                     prediction_type  = config.prediction_type,
                     layers_params    = 'standard',
                     kw_features_size = 100,
                     feedback_size    = 5,
                     data_sample      = data_sample,
                     device           = device,
                     verbose          = config.verbose)
    model = model.to(device)
    assess_num_params(model=model, data_sample=data_sample, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer.zero_grad()
    
    if config.prediction_type==GNNAgent.NEXT_FEEDBACK:
        weight = CorpusGraphDataset.get_class_weights(loader=train_loader, mode=config.exp_split_mode, dataset_name=config.exp_dataset_name).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    elif config.prediction_type==GNNAgent.NEXT_RECO:
        # assert weight[1] > weight[0]
        # pos_weight = torch.FloatTensor([weight[1]/weight[0]]).to(device)
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'prediction_type "{config.prediction_type}" not supported.')
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', threshold_mode='abs',
                                  factor=config.scheduler_factor, patience=config.scheduler_patience, threshold=config.scheduler_threshold,
                                  verbose=True)
    
    log_cf_mx_img = config.log_cf_mx_img if hasattr(config, 'log_cf_mx_img') else False
    
    first_idx = 0 if config.init_eval else 1
    for epoch in range(first_idx, config.num_epochs+1):
        build_cf_mx_img = epoch==config.num_epochs if log_cf_mx_img else False
        train_metrics_dict, train_cf_mx = train_eval_epoch(model=model, criterion=criterion, device=device, build_cf_mx_img=build_cf_mx_img, verbose=config.verbose,
                                                           loader=train_loader,
                                                           optimizer=optimizer,
                                                           is_train= epoch!=0,
                                                           config=config,
                                                           use_wandb=use_wandb)
        eval_metrics_dict, eval_cf_mx = train_eval_epoch(model=model, criterion=criterion, device=device, build_cf_mx_img=build_cf_mx_img, verbose=config.verbose,
                                                        loader=eval_loader,
                                                        optimizer=None,
                                                        is_train=False,
                                                        config=config,
                                                        use_wandb=use_wandb)
        # log results
        train_metrics_dict = { exputils.add_prefix(metric,TRAIN):val for metric,val in train_metrics_dict.items()}
        eval_metrics_dict  = { exputils.add_prefix(metric,EVAL):val for metric,val in eval_metrics_dict.items()}
        results_dict = {**train_metrics_dict, **eval_metrics_dict}
        if config.verbose >= 1: print_epoch_results(epoch=epoch, results_dict=results_dict)
        if use_wandb:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'epoch':epoch, 'lr':current_lr, **results_dict})
        
        # scheduler
        train_loss_metric = exputils.add_prefix(LOSS,TRAIN)
        if train_loss_metric in train_metrics_dict:
            scheduler.step(train_metrics_dict[train_loss_metric])
    
    return model, train_loader, eval_loader, results_dict, train_cf_mx, eval_cf_mx


def train_eval_epoch(model:GNNAgent, loader:DataLoader, criterion:torch.nn.CrossEntropyLoss, optimizer:torch.optim.Optimizer, device:str, is_train:bool, config, build_cf_mx_img:bool=False, use_wandb:bool=False, verbose:int=1):
    if loader is None:
        return {}, None
    
    if is_train:
        if verbose >= 1: print("\nTRAIN")
        assert optimizer is not None
        model.train()
        torch.set_grad_enabled(True)
    else:
        if verbose >= 1: print("\nEVAL")
        model.eval()
        torch.set_grad_enabled(False)
    
    is_expert_policy = model.prediction_type==GNNAgent.NEXT_RECO
    
    total_loss, total_correct, total_num_samples = 0, 0, 0
    all_preds, all_labels, all_right_recos_probas  = [], [], []
    iter = 0
    for data_batch in loader:
        data_batch = data_batch.to(device)
        labels = data_batch['doc'].y.float()
        num_samples = len(data_batch) if is_expert_policy else int(labels.shape[0])

        out, _ = model(data_batch.x_dict, data_batch.edge_index_dict, corpus_graph_list=data_batch["corpus_graph"])

        # new
        if is_expert_policy:
            out = torch.squeeze(out)
            ptr = data_batch['doc'].ptr
            add_padding = True
            add_softmax = config.add_softmax
            if add_padding:
                num_docs = [data.num_docs for data in data_batch.to_data_list()]
                num_docs_max = max(num_docs)
                logits = [torch.cat([out[start:end]   , torch.full((num_docs_max-(end-start),), -float('inf')).to(device)]) for start, end in zip(ptr[:-1], ptr[1:])]
                # labels = torch.argmax(torch.tensor([labels[start:end] for start, end in zip(ptr[:-1], ptr[1:])]), dim=1)
                labels = torch.tensor([torch.argmax(labels[start:end]).item() for start, end in zip(ptr[:-1], ptr[1:])]).to(device)

                if add_softmax:
                    logits = [F.softmax(logit, dim=0) for logit in logits]
            else:
                if add_softmax:
                    logits = [F.softmax(out[start:end], dim=0) for start, end in zip(ptr[:-1], ptr[1:])]
                else:
                    logits = out
            
            logits = torch.stack(logits, dim=0)
        else:
            logits = out
        
        loss = criterion(logits, labels)
        
        if is_expert_policy:
            # probabilities_batch = torch.sigmoid(logits)
            # preds = (probabilities_batch >= 0.5).float()
            probas = F.softmax(logits, dim=1) if not add_softmax else logits
            preds = probas.argmax(dim=1)
            right_recos_probas = torch.tensor([ probas[i, labels[i]].item() for i in range(labels.shape[0])])
        else:
            preds = logits.argmax(dim=1)

        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if is_expert_policy:
            all_right_recos_probas.extend(right_recos_probas.cpu().numpy())

        weighted_loss, acc, num_correct, num_samples = compute_metrics(preds=preds.cpu(), labels=labels.cpu(), loss=loss.cpu(), num_samples=num_samples, is_expert_policy=is_expert_policy)
        total_loss        += weighted_loss
        total_correct     += num_correct
        total_num_samples += num_samples
        iter += 1

        if device.type=='cuda':
            p_right_reco_info = f", p_right_reco = {round(torch.mean(right_recos_probas).item(),3):>5}" if is_expert_policy else ""
            if verbose >=4 and iter%10==0: print(f"iter {iter}: acc = {acc:.4f}, loss = {weighted_loss:.4f} {p_right_reco_info} (gpu usage: {exputils.get_gpu_usage(verbose=False)})")
        else:
            if verbose >=4 and iter%10==0: print(f"iter {iter}: acc = {acc:.4f}, loss = {weighted_loss:.4f}")
    
    acc = total_correct / total_num_samples
    cumul_loss = total_loss / total_num_samples
    
    if is_expert_policy:
        right_recos_probas_mean = np.mean(all_right_recos_probas)
    
    precision_macro, recall_macro, f1_macro, true_sum_macro = precision_recall_fscore_support(y_true=all_labels, y_pred=all_preds, beta=1, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_vector   = f1_score(all_labels, all_preds, average=None)

    metrics_dict = {LOSS:cumul_loss, ACC:acc, PRECISION_MACRO:precision_macro, RECALL_MACRO:recall_macro, F1_MACRO:f1_macro, F1_WEIGHTED:f1_weighted, F1_0:f1_vector[0], F1_1:f1_vector[1], F1_2:f1_vector[2]}
    if is_expert_policy:
        metrics_dict.update({RIGHT_RECO_PROBA:right_recos_probas_mean})
    # else:
    #     metrics_dict.update({PRECISION_MACRO:precision_macro, RECALL_MACRO:recall_macro, F1_MACRO:f1_macro, F1_WEIGHTED:f1_weighted, F1_0:f1_vector[0], F1_1:f1_vector[1], F1_2:f1_vector[2]})

    if use_wandb and not is_expert_policy:
        import wandb
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds, class_names=CLASS_NAMES_FEEDBACK)})
    if build_cf_mx_img and not is_expert_policy:
        cf_matrix = confusion_matrix(all_labels, all_preds)
        if use_wandb:
            cf_mx_plot = plot_confusion_mx(cf_matrix=cf_matrix, path=None, save=False)
            wandb.log({"conf_mat_plot": cf_mx_plot})
    else:
        cf_matrix = None

    return metrics_dict, cf_matrix


def assess_num_params(model, data_sample, device):
    data_sample = data_sample.to(device)
    model(data_sample.x_dict, data_sample.edge_index_dict, corpus_graph_list=data_sample["corpus_graph"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal number of parameters: {total_params}\n')


def print_epoch_results(epoch, results_dict):
    d = {}
    for key,val in results_dict.items():
        d[key] = nan if val is None else val
    # d = SimpleNamespace(**d)
    text = f'\nEpoch: {epoch:03d}'
    for metric in [LOSS, F1_MACRO, ACC]:
        text += ' | '
        train_metric_key = exputils.add_prefix(metric,TRAIN)
        eval_metric_key  = exputils.add_prefix(metric,EVAL)
        train_val = d[train_metric_key]
        if eval_metric_key in d.keys():
            eval_val  = d[eval_metric_key]
            text += f'{metric} (tr/ev):{train_val:.4f}/{eval_val:.4f}'
        else:
            text += f'{metric} (tr):{train_val:.4f}'
    print(text)
    return text

def plot_confusion_mx(cf_matrix, path:str=None, save:bool=True):
    class_names = [Feedback.ID2MEANING[feedback_id] for feedback_id in Feedback.FEEDBACKS if feedback_id != Feedback.NOT_VISITED]
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_names],
                        columns = [i for i in class_names])
    fig = plt.figure(figsize = (12,7))
    sn.heatmap(data=df_cm, cmap="Blues", annot=True)
    if save:
        assert path is not None
        if not '.png' in path: path += '.png'
        plt.savefig(path)
    return fig


def compute_metrics(preds:torch.Tensor, labels:torch.Tensor, loss, num_samples:int, is_expert_policy:bool):
    weighted_loss = float(loss) * num_samples    
    correct = preds == labels  # Check against ground-truth labels.
    num_correct = int(correct.sum())
    acc = num_correct/num_samples  # Derive ratio of correct predictions.
    return weighted_loss, acc, num_correct, num_samples


def save_sl_training(config_dict:dict, results_dict:dict=None, model=None, sweep_id=None):
    datetime = exputils.utils.current_datetime()
    
    if config_dict['exp_dataset_name']=='expert_policy':
        results_folder_path = os.path.join(PathManager.EXPERT_RESULTS, datetime)
    elif config_dict['exp_dataset_name']=='first_dataset':
        results_folder_path = os.path.join(PathManager.SL_RESULTS, datetime)
    else:
        folder_name = config_dict['exp_dataset_name'] if (isinstance(config_dict['exp_dataset_name'],str) and config_dict['exp_dataset_name'] != "") else 'default'
        results_folder_path = os.path.join(PathManager.RESULTS, folder_name, datetime)
    os.makedirs(results_folder_path)

    results_file_path = os.path.join(results_folder_path, "perfs.json")
    config_file_path  = os.path.join(results_folder_path, "config.json")
    infos_path        = os.path.join(results_folder_path, 'infos.json')
    model_path        = os.path.join(results_folder_path, 'model.pt')
    cf_train_path     = os.path.join(results_folder_path, 'confusion_train')
    cf_eval_path      = os.path.join(results_folder_path, 'confusion_eval' )
    
    infos_dict = dict(sweep_id=sweep_id)
    
    exputils.utils.save_dict_as_json(data_dict=config_dict, file_path=config_file_path)
    exputils.utils.save_dict_as_json(data_dict=infos_dict, file_path=infos_path)
    data_dict = {result_key:result_val for result_key,result_val in results_dict.items() if not result_key in {'train_cf_mx', 'eval_cf_mx'}}
    if results_dict is not None: exputils.utils.save_dict_as_json(data_dict=data_dict, file_path=results_file_path)
    if model        is not None: torch.save(model, model_path)
    if 'train_cf_mx' in results_dict.keys(): plot_confusion_mx(cf_matrix=results_dict['train_cf_mx'], path=cf_train_path, save=True)
    if 'eval_cf_mx'  in results_dict.keys(): plot_confusion_mx(cf_matrix=results_dict['eval_cf_mx'] , path=cf_eval_path,  save=True)

    print(f'\nSaved model and results in {results_folder_path}')



def set_wandb_params(use_wandb:bool, names_dict:dict):
    if not use_wandb:
        return None, None
    wandb_names = dict(entity='rl4edu', project='pre-trained-reco-system-project')
    if not 'group' in names_dict.keys(): names_dict['group'] = DEFAULT_WANDB_GROUP_NAME
    metric_goal = {'name':'eval/loss','goal':'minimize'}
    return wandb_names, metric_goal

def sweep_trainer(config_dict=None):
    with wandb.init(config=config_dict) as run:
        train_func(config=wandb.config, use_wandb=True, run=run)


def preprocess_quick_test(config):
    return config


if __name__=='__main__':

    verbose  = 4

    quick_test = False
    use_wandb  = True
    use_sweep  = True
    
    init_eval = True
    save_perfs_and_model = False

    exp_cfg = dict(
        exp_id    = 'basic_training',
        seed      = [0],
        device    = 'default',
        init_eval = init_eval,
        verbose   = verbose
    )
    dataset_cfg = dict(
        exp_dataset_name = "expert_policy", # first_dataset
        prediction_type  = "next_reco",
        exp_split_mode   = "default1",
    )
    gnn_cfg = dict(
        gnn_arch         = "transformer16",
        feedback_arch    = "linear2",
        gnn_act          = "elu",
        feedback_act     = "relu",
        hidden_dim       = 128,
        heads            = [4,8],
        f_dropout_mode   = 'all_except_last',
        gnn_dropout_mode = 'last_only',
        dropout_features = 0.2,
        dropout_conv     = [0.6, 0.4],
        dropout_dense    = 0.2,
        dropout_f        = [0.4, 0.2],
        edge_dropout     = 0.1,
        concat           = [False, True],
        beta             = True,
        aggr             = ['add', 'mean']
    )
    training_cfg = dict(
        batch_size_train = 16, # 128
        batch_size_valid = 16, # 256
        add_softmax      = [False],
        optimizer_name   = 'adam',
        lr               = [0.01, 0.001],
        num_epochs       = 5,
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
    config = {**exp_cfg, **dataset_cfg, **gnn_cfg, **training_cfg, **data_cfg, **scheduler_cfg}


    arguments = exputils.retrieve_arguments()
    mode, names_dict, cluster_name = exputils.set_experiment_mode(arguments=arguments)
    wandb_names, metric_goal = set_wandb_params(use_wandb=use_wandb, names_dict=names_dict)
    names_dict = {**names_dict, **wandb_names} if use_wandb else None
    
    if mode=="generate_slurm":
        exputils.generate_slurm(SLURM_PATH=PathManager.SLURM, cluster_name=cluster_name, SlurmGenerator_cls=CustomSlurmGenerator)
    elif mode=="cluster":
        exputils.run_in_cluster_mode(train_func=train_func, CONFIGS_PATH=PathManager.CONFIGS, SYNC_WANDB_PATH=PathManager.SYNC_WANDB, names_dict=names_dict)
    elif mode=="standard":
        exputils.run_in_standard_mode(config=config, train_func=train_func,
                                      quick_test=quick_test, use_sweep=use_sweep, use_wandb=use_wandb, is_offline=False,
                                      SYNC_WANDB_PATH=PathManager.SYNC_WANDB, names_dict=names_dict, metric_goal=metric_goal,
                                      sweep_trainer=sweep_trainer, preprocess_quick_test_func=preprocess_quick_test,
                                      wandb_method="grid")
    else:
        raise ValueError(f'Mode {mode} not supported.')

        