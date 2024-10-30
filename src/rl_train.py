import os

from pprint import pprint as pprint
from types import SimpleNamespace

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from tianshou.policy import PPOPolicy, NPGPolicy, PGPolicy, BasePolicy

import wandb

import experiments_utils as exputils

from .custom_slurm_generator import CustomSlurmGenerator

from . import PathManager
from .student_simulation import CorpusGraphDataset, Population
from .rl_modules.env import AdaptiveLearningEnv
from . import utils

from .model import GNNAgent, MLPAgent

from .rl_modules import CustomPGPolicy, CustomOnpolicyTrainer, CustomActor, CustomCritic, CustomWandbLogger
from .rl_modules.custom_onpolicy_trainer import custom_onpolicy_trainer


DEFAULT_WANDB_GROUP_NAME = 'rl_training'


def create_env_with_params(config, is_eval:bool=False, verbose:int=0):
    pretrain_split_mode = maybe_load_pretrain_stats(config)
    training_mode = 'non_linear_finetuning' if is_eval and config.non_linear_eval_env else config.training_mode
    split_mode    = 'none'                  if is_eval and config.non_linear_eval_env else config.exp_split_mode
    corpus_graph_list = CorpusGraphDataset.generate_train_rl_corpus_graphs(training_mode=training_mode,
                                                                            kw_normalization=config.kw_normalization,
                                                                            split_mode=split_mode,
                                                                            pretrain_split_mode=pretrain_split_mode)
    prior_knowledge_distrib = "zero" if is_eval and config.non_linear_eval_env else config.prior_knowledge_distrib
    population = Population(prior_knowledge_distrib  = prior_knowledge_distrib,
                            prereq_distrib           = config.prereq_distrib,
                            prior_background_distrib = config.prior_background_distrib)
    obs_manager = "bassen" if config.model_type=='bassen' else 'default'
    env = AdaptiveLearningEnv(corpus_graphs_list=corpus_graph_list,
                              obs_manager  = obs_manager,
                              population   = population,
                              human_mode   = False,
                              horizon_mode = "target_kc",
                              horizon_coef = config.horizon_coef,
                              render_mode  = None,
                              is_eval      = is_eval,
                              verbose      = verbose)
    # other operations such as env.seed(np.random.choice(10))
    def create_env():
        return env
    return create_env


def maybe_load_pretrain_stats(config):
    if (config.load_model is not None) and (config.load_model != 'none'):
        pretrain_config_path = os.path.join(PathManager.SAVED_MODELS, config.load_model, 'config.json')
        pretrain_config = utils.load_json_file(file_path=pretrain_config_path)
        return pretrain_config['exp_split_mode']
    else:
        return None


def maybe_preprocess_config(config):
    # num_students = max_epoch * step_per_epoch * episode_per_collect
    if hasattr(config, 'num_students') and config.num_students is not None:
        if not hasattr(config, 'max_epoch'):
            assert config.num_students%(config.step_per_epoch*config.episode_per_collect)==0
            max_epoch = int(config.num_students/(config.step_per_epoch*config.episode_per_collect))
            config.max_epoch = max_epoch
    return config


def train_func(config, use_wandb:bool, run=None):
    global BEGIN_DATETIME
    global CONFIG
    device = exputils.preprocess_training(config=config, seed=config.seed, device=config.device)
    exputils.maybe_define_wandb_metrics(loss_metrics=CustomWandbLogger.LOSS_METRICS, score_metrics=CustomWandbLogger.SCORE_METRICS, stages=CustomWandbLogger.STAGES,
                                        use_wandb=use_wandb, custom_step_metric=CustomWandbLogger.EP_COUNT)

    config = maybe_preprocess_config(config)
    check_config_consistency(config)

    if run is not None:
        sweep_id, name, group = run.sweep_id, run.name, run.group
    else:
        sweep_id, name, group = 'none', 'none', 'none'
    BEGIN_DATETIME = utils.current_datetime() + "_" + sweep_id + "_" + name + "_" + group
    CONFIG = config

    train_envs = DummyVectorEnv([create_env_with_params(config, verbose=config.verbose) for _ in range(config.num_envs)])
    eval_envs  = DummyVectorEnv([create_env_with_params(config, is_eval=True, verbose=config.verbose) for _ in range(1)])
    
    if config.model_type=='bassen':
        bassen_cfg_path = os.path.join(PathManager.BASSEN_CONFIGS, config.bassen_cfg)
        bassen_cfg_dict:dict = utils.load_yaml_file(filepath=bassen_cfg_path)
        if use_wandb: wandb.config.update({f'bassen_{key}':val for key,val in bassen_cfg_dict.items()})
        bassen_cfg = SimpleNamespace(**bassen_cfg_dict)
        obs = train_envs.reset()[0][0]
        data_sample = CustomActor.preprocess_batch([obs])
        num_actions = data_sample.x_dict['feedback'].shape[1]  // 5
        input_dim   = data_sample.x_dict['feedback'].shape[1]
        model_actor  = MLPAgent(input_dim=input_dim, hidden_sizes=bassen_cfg.hidden_sizes_actor, device=device, verbose=config.verbose)
        model_critic = MLPAgent(input_dim=input_dim, hidden_sizes=bassen_cfg.hidden_sizes_critic, device=device, verbose=config.verbose)
        actor        = CustomActor(model=model_actor, keep_sl_head=None, freeze_layers=None, device=device, num_actions=num_actions)
        critic       = CustomCritic(model=model_critic, device=device)
        actor_critic = ActorCritic(actor, critic).to(device)
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.lr)
        dist = torch.distributions.Categorical
        policy = PPOPolicy(actor=actor,
                           critic=critic,
                           optim=optimizer,
                           dist_fn=dist,
                           ent_coef=config.ent_coef,
                           eps_clip=config.eps_clip)
        model = None # just for the final return


    elif config.model_type=='GNNAgent':
        if (config.load_model is not None) and (config.load_model != 'none'):
            path = os.path.join(PathManager.SAVED_MODELS, config.load_model, 'model.pt')
            model = torch.load(path)
            assert isinstance(model, GNNAgent)
            model.set_dropout_to_zero()
        else:
            obs = train_envs.reset()[0][0]
            data_sample = CustomActor.preprocess_batch([obs])
            model = GNNAgent(gnn_arch        = config.gnn_arch,
                            feedback_arch    = config.feedback_arch,
                            gnn_act          = config.gnn_act,
                            feedback_act     = config.feedback_act,
                            hidden_dim       = config.hidden_dim,
                            heads            = config.heads,
                            f_dropout_mode   = 'none',
                            gnn_dropout_mode = 'none',
                            dropout_features = 0,
                            dropout_conv     = 0,
                            dropout_dense    = 0,
                            dropout_f        = 0,
                            edge_dropout     = 0,
                            aggr             = config.aggr,
                            concat           = config.concat,
                            beta             = config.beta,
                            layers_params    = 'standard',
                            prediction_type  = 'next_reco',
                            kw_features_size = 100,
                            feedback_size    = 5,
                            data_sample      = data_sample,
                            device           = device,
                            verbose          = config.verbose)

        if config.policy_name=="PGPolicy":
            actor     = CustomActor(model=model, keep_sl_head=config.keep_sl_head, freeze_layers=config.freeze_layers, device=device).to(device)
            optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
            if config.use_scheduler:
                scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold_mode='abs',
                                            factor=config.scheduler_factor, patience=config.scheduler_patience, threshold=config.scheduler_threshold,
                                            verbose=True)
            else:
                scheduler = None
            dist = torch.distributions.Categorical
            policy = CustomPGPolicy(model=actor, optim=optimizer, dist_fn=dist, action_space=None,
                                    discount_factor = config.discount_factor,
                                    reward_normalization = config.reward_normalization,
                                    action_scaling=False,
                                    lr_scheduler=scheduler,
                                    deterministic_eval=False,
                                    ent_coef=config.ent_coef)
        else:
            raise ValueError(f'Policy {config.policy_name} not supported.')
    else:
        raise ValueError(f'Model "{config.model_type}" not supported.')
    

    # collector
    # assert config.multiply_factor >= 1
    # step_per_collect = config.episode_per_collect * 
    # total_size = int(step_per_collect * config.multiply_factor)
    total_size = 20000
    replayBuffer    = VectorReplayBuffer(total_size=total_size, buffer_num=len(train_envs))
    train_collector = Collector(policy=policy, env=train_envs, buffer=replayBuffer)
    eval_collector  = Collector(policy=policy, env=eval_envs)
    
    logger = CustomWandbLogger(optimizer = optimizer,
                               train_interval_ep  = 1,
                               test_interval_ep   = 1,
                               update_interval_ep = 1,
                               use_wandb=use_wandb)
    
    save_best_fn_param = save_best_fn if config.save_best_fn else None
    train_results = custom_onpolicy_trainer(policy=policy,
                                            train_collector=train_collector,
                                            test_collector=eval_collector,
                                            max_epoch=config.max_epoch,
                                            step_per_epoch=config.step_per_epoch,
                                            repeat_per_collect=config.repeat_per_collect,
                                            episode_per_test=config.episode_per_test,
                                            batch_size=config.batch_size,
                                            step_per_collect=config.step_per_collect,
                                            episode_per_collect=config.episode_per_collect,
                                            save_best_fn = save_best_fn_param,
                                            logger=logger)
    
    if hasattr(config, 'save_model') and config.save_model:
        if run is not None:
            sweep_id, name, group = run.sweep_id, run.name, run.group
        else:
            sweep_id, name, group = 'none', 'none', 'none'
        save_rl_training(config_dict=dict(config), results_dict=train_results, model=model, sweep_id=sweep_id, name=name, group=group)
    
    return train_results, model

def save_best_fn(policy:CustomPGPolicy):
    global BEGIN_DATETIME
    global CONFIG
    results_folder_path = os.path.join(PathManager.RL_RESULTS, 'current_best', BEGIN_DATETIME)
    os.makedirs(results_folder_path, exist_ok=True)
    model_path       = os.path.join(results_folder_path, 'model.pt')
    config_file_path = os.path.join(results_folder_path, "config.json")
    print(f'\nNew best test reward. Saving model in {model_path}\n')
    torch.save(policy.actor.model, model_path)
    utils.save_dict_as_json(data_dict=dict(CONFIG), file_path=config_file_path)


def check_config_consistency(config):
    if hasattr(config, 'num_students') and config.num_students is not None:
        wanted_num_students = config.num_students
        estimated_num_of_students = config.episode_per_collect * config.step_per_epoch * config.max_epoch
        if wanted_num_students != "none":
            assert config.num_students == estimated_num_of_students, f'Inconsistent num of students: {wanted_num_students} (wanted) vs. {estimated_num_of_students} (actual).'
        assert (not hasattr(config,'step_per_collect')) or config.step_per_collect is None
    else:
        assert hasattr(config, 'step_per_collect') and config.step_per_collect is not None
        assert (not hasattr(config,'episode_per_collect')) or config.episode_per_collect is None


def save_rl_training(config_dict:dict, results_dict:dict=None, model=None, sweep_id='none', name='none', group='none'):
    datetime = utils.current_datetime()
    results_folder_path = os.path.join(PathManager.RL_RESULTS, datetime)
    os.makedirs(results_folder_path)
    
    results_file_path = os.path.join(results_folder_path, "perfs.json")
    config_file_path  = os.path.join(results_folder_path, "config.json")
    infos_path        = os.path.join(results_folder_path, 'infos.json')
    model_path        = os.path.join(results_folder_path, 'model.pt')
    
    infos_dict = dict(sweep_id=sweep_id, name=name, group=group)
    
    utils.save_dict_as_json(data_dict=config_dict, file_path=config_file_path)
    utils.save_dict_as_json(data_dict=infos_dict, file_path=infos_path)
    if results_dict is not None: utils.save_dict_as_json(data_dict=results_dict, file_path=results_file_path)
    if model        is not None: torch.save(model, model_path)   


def set_wandb_params(use_wandb:bool, names_dict:dict):
    if not use_wandb:
        return None, None
    wandb_names = dict(entity='rl4edu', project='pre-trained-reco-system-project')
    if not 'group' in names_dict.keys(): names_dict['group'] = DEFAULT_WANDB_GROUP_NAME
    metric_goal = {'name':f'{CustomWandbLogger.TEST}/{CustomWandbLogger.REW}','goal':'maximize'}
    return wandb_names, metric_goal

def sweep_trainer(config_dict=None):
    with wandb.init(config=config_dict) as run:
        train_func(config=wandb.config, use_wandb=True, run=run)


def preprocess_quick_test(config):
    return config


if __name__=="__main__":

    verbose  = 4
    
    quick_test = False
    use_wandb  = False
    use_sweep  = False


    exp_cfg = dict(
        exp_id        = 'rl_finetuning1',
        training_mode = 'non_linear_finetuning', # 'non_linear_finetuning' or 'pretraining'
        model_type    = "bassen",                # GNNAgent, Bassen
        bassen_cfg    = 'config0',
        seed          = [0],
        device        = 'default',
        verbose       = verbose,
        version       = 'sept2024'
    )
    data_cfg = dict(
        kw_normalization         = 'none',        # 'z-score' or 'unit' or 'none'
        exp_split_mode           = 'none', # 'none' or ...
        prior_knowledge_distrib  = 'zero',  # 'zero', 'uniform', 'decreasing_exponential'
        prereq_distrib           = 'uniform',
        prior_background_distrib = 'binomial',        # 'binomial' or 'none' for linear corpora
        horizon_coef             = 1.5,
        num_envs                 = 1,
        save_model               = False,
        non_linear_eval_env      = False
    )
    gnn_cfg = dict(
        # load_model       = 'pretrain/sl/pretrain_expert_full1', # 'none', 'pretrain/sl/pretrain_full1', 'pretrain/sl/pretrain_expert_full1', 'pretrain/rl/pretrain_full1'
        load_model       = 'none',
        freeze_layers    = "none", # 'none', 'all', 'all_except_last_gnn_module'
        keep_sl_head     = True,
        gnn_arch         = "transformer16",
        feedback_arch    = "linear2",
        gnn_act          = "elu",
        feedback_act     = "relu",
        hidden_dim       = 256,
        heads            = 4,
        concat           = False,
        beta             = True,
        aggr             = 'mean'
    )
    exp_cfg_trainer = dict(
        num_students        = 50, # num_students = max_epoch * step_per_epoch * episode_per_collect
        step_per_epoch      = 1,
        episode_per_collect = 5,   # 1
        step_per_collect    = None,
        episode_per_test    = 20,  # 10
    )
    rl_cfg = dict(
        policy_name = "PGPolicy"
    )
    optimizer_cfg = dict(
        lr = 0.0005
    )
    trainer_cfg = dict(
        repeat_per_collect = 15,
        batch_size         = 16
    )
    PGPolicy_cfg = dict(
        discount_factor      = 0,
        ent_coef             = 0.,
        reward_normalization = [True, False],
        use_scheduler        = True,
        scheduler_factor     = 0.5,
        scheduler_patience   = 10,
        scheduler_threshold  = 0.5
    )
    PPO_cfg = dict(
        eps_clip = 0.2
    )
    replay_buffer_cfg = dict(
        multiply_factor =  [1.]# [2., 1.5, 1.]
    )
    config = {**exp_cfg, **data_cfg, **exp_cfg_trainer, **gnn_cfg, **rl_cfg, **optimizer_cfg, **trainer_cfg, **PGPolicy_cfg, **PPO_cfg, **replay_buffer_cfg}

    arguments = exputils.retrieve_arguments()
    mode, names_dict, cluster_name = exputils.set_experiment_mode(arguments=arguments)
    wandb_names, metric_goal = set_wandb_params(use_wandb=use_wandb, names_dict=names_dict)
    names_dict = {**names_dict, **wandb_names} if use_wandb else None
    
    if mode=="generate_slurm":
        filename = os.path.basename(__file__)
        filename = filename.split('.py')[0]
        exputils.generate_slurm(config=config, cluster_name=cluster_name, filename=filename, SlurmGenerator_cls=CustomSlurmGenerator)
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

    # if use_wandb:
    #     import wandb
    #     wandb_params = dict(entity="rl4edu", project="pre-trained-reco-system-project", group="rl_finetuning", dir="./logs")
    #     run = wandb.init(config=config, sync_tensorboard=True, **wandb_params)
    
    # config_object = SimpleNamespace(**config)
    # train_func(config=config_object, num_envs=None, device=None, verbose=verbose, use_wandb=use_wandb)
