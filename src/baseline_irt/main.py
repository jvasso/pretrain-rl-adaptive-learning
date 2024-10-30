from typing import Dict, List
from types import SimpleNamespace

import numpy as np
import wandb
import random

import experiments_utils as exputils

from ..student_simulation import Population, Student, CorpusGraph
from ..student_simulation.types import Interaction, Feedback, InteractionsHistory, NoneInteraction

from ..rl_modules import CustomWandbLogger

from ..path_manager import PathManager

from .lin_ts import LinearTS
from . import utils

DEFAULT_WANDB_GROUP_NAME = 'irt_baseline'


def random_reco(doc_id2feedback) -> int:
    doc_list = list(doc_id2feedback.keys())
    return random.choice(doc_list)

def calculate_contexts(interactions_history) -> list:
    ctx_list = [None for _ in interactions_history.doc2last_interaction]

    for doc_idx, doc_feedback in interactions_history.doc2last_interaction.items():
        if isinstance(doc_feedback, NoneInteraction):
            ctx_list[doc_idx] = np.array([0, 0, 0, 1])
        else:
            ctx_list[doc_idx] = np.array([0, 0, 0, 1])
            ctx_list[doc_idx][doc_feedback.get_label()] = 1.

    return ctx_list


def run_session(config, use_wandb, run=None):
    utils.set_all_seeds(seed=config.seed)

    population = Population(prior_knowledge_distrib  = config.prior_knowledge_distrib,
                            prereq_distrib           = config.prereq_distrib,
                            prior_background_distrib = config.prior_background_distrib,
                            feedback_mode            = 'default')
    corpus_graph = CorpusGraph(corpus_name='intro_to_ml', corpus_group='hand_made', corpus_type='non_linear', load_minimal_data=True)
    target_kc_list = [kc for kc in corpus_graph.kc_list if kc.is_regular() and kc.level=='1']
    horizon = len(target_kc_list)

    total_learning_gains_list = []
    num_docs_learned_list = []

    # Initialize bandit
    agent = LinearTS(4, len(corpus_graph.doc_list), {})
    
    student_count = 0
    for student in population.sample_students_iter(corpus_graph=corpus_graph, num=config.num_students): # sampling student from population
        student_count += 1
        if config.verbose >= 2: print(f'Student {student_count}')

        total_learning_gains = 0 # each document contains Knowledge Components with associated values. learning gains = sum of values
        num_docs_learned = 0

        step_count = 0
        terminated = False
        while not terminated:
            
            # some useful info about student knowledge
            knowledge_map:Dict[int,bool] = student.get_current_knowledge_docs() # doc_id -> bool (student knows or not)
            doc_id2feedback:Dict[int,Feedback] = student.doc_id2feedback(feedback_mode='default', return_tensor=True) # doc_id -> student last feedback (including 'not visited' feedback)
            interactions_history:InteractionsHistory = student.interactions_history
            
            # recommend document
            ctx_list = calculate_contexts(interactions_history)
            agent_action = agent.queryContexts(ctx_list)
            agent_action_id = list(doc_id2feedback.keys())[agent_action]
            
            # student interacts with document
            interaction:Interaction = student.interact_with_doc_id(doc_id=agent_action_id, time_step=step_count)
            has_learned = interaction.has_learned()
            learning_gains = interaction.feedback.get_learning_score()

            agent.updateContext(ctx_list, agent_action, learning_gains)

            if config.verbose >= 2: print(f'Step {step_count:>2} | action: {agent_action:>2} | has learned: {"true" if has_learned else "false":>5} | learning gains: {learning_gains}')

            total_learning_gains += learning_gains
            if has_learned: num_docs_learned += 1

            step_count += 1
            terminated, cause = utils.is_terminated(student=student, step_count=step_count, horizon=horizon, target_kc_list=target_kc_list)
        
        if config.verbose >= 2: print(f'Terminated (cause: {cause}). Total learning gains: {total_learning_gains}\n')

        total_learning_gains_list.append(total_learning_gains)
        num_docs_learned_list.append(num_docs_learned)

        # logging
        if student_count%config.log_freq==0:
            num_samples = config.log_freq
            learning_gains_moving_avg = np.mean(total_learning_gains_list[-num_samples:])
            docs_learned_moving_avg   = np.mean(num_docs_learned_list[-num_samples:])
            if config.verbose >= 1: print(f'• {student_count:>2} students | learning gains: {round(learning_gains_moving_avg,2)} | num docs learned: {round(docs_learned_moving_avg,2)}')
            
            if use_wandb:
                # wandb.log({'student_count':student_count, 'learning_gains':learning_gains_moving_avg, 'docs_learned':docs_learned_moving_avg})
                 wandb.log({'episode':student_count, f'{CustomWandbLogger.TEST}/{CustomWandbLogger.REW}':learning_gains_moving_avg, 'docs_learned':docs_learned_moving_avg})


def set_wandb_params(use_wandb:bool, names_dict:dict):
    if not use_wandb:
        return None, None
    wandb_names = dict(entity='rl4edu', project='pre-trained-reco-system-project')
    if not 'group' in names_dict.keys(): names_dict['group'] = DEFAULT_WANDB_GROUP_NAME
    metric_goal = {'name':f'{CustomWandbLogger.TEST}/{CustomWandbLogger.REW}','goal':'maximize'}
    return wandb_names, metric_goal


def sweep_trainer(config_dict=None):
    with wandb.init(config=config_dict) as run:
        run_session(config=wandb.config, use_wandb=True, run=run)


def preprocess_quick_test(config):
    return config


if __name__=="__main__":

    verbose = 1
    
    quick_test = False
    use_wandb  = True
    use_sweep  = True
    
    exp_cfg = dict(
        exp_id  = 'irt_baseline',
        seed    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
        device  = 'default',
        version = 'sept2024',
        verbose = verbose
    )
    data_cfg = dict(
        num_students             = 50,
        prior_knowledge_distrib  = ['zero', 'uniform', 'decreasing_exponential'], # zero/uniform/decreasing_exponential
        prereq_distrib           = 'uniform', # keep like this
        prior_background_distrib = 'binomial' # keep like this
    )
    log_cfg = dict(
        log_freq = 5
    )
    config = {**exp_cfg, **data_cfg, **log_cfg}
    
    # if use_wandb:
    #     run = wandb.init(config=config, sync_tensorboard=False, **wandb_params)
    # run_session(config=config, use_wandb=use_wandb)

    names_dict = {}
    wandb_params = dict(entity='rl4edu', project='pre-trained-reco-system-project')
    wandb_names, metric_goal = set_wandb_params(use_wandb=use_wandb, names_dict=names_dict)
    names_dict = {**names_dict, **wandb_names} 
    
    exputils.run_in_standard_mode(config=config, train_func=run_session,
                                  quick_test=quick_test, use_sweep=use_sweep, use_wandb=use_wandb, is_offline=False,
                                  SYNC_WANDB_PATH=PathManager.SYNC_WANDB, names_dict=names_dict, metric_goal=metric_goal,
                                  sweep_trainer=sweep_trainer, preprocess_quick_test_func=preprocess_quick_test,
                                  wandb_method="grid")

