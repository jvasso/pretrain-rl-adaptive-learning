import os
from typing import List, Union

from . import utils

class PathManager:
    
    PROJECT = "."

    SLURM        = os.path.join(PROJECT, 'slurm')
    CONFIGS      = os.path.join(PROJECT, "configs")
    LOGFILES     = os.path.join(PROJECT, "logfiles")
    SYNC_WANDB   = os.path.join(CONFIGS, "sync_wandb.sh")
    
    SAVED_MODELS = os.path.join(PROJECT, "saved_models")
    
    RESULTS = os.path.join(PROJECT, "results")
    SL_RESULTS     = os.path.join(RESULTS, "sl_results")
    RL_RESULTS     = os.path.join(RESULTS, "rl_results")
    EXPERT_RESULTS = os.path.join(RESULTS, "expert_results")

    DATA    = os.path.join(PROJECT, "data")
    CORPORA = os.path.join(DATA, "corpora")
    EVAL_CORPORA = os.path.join(DATA, "eval_corpora")
    SRC = os.path.join(PROJECT, "src")
    
    LINEAR_CORPUS_TYPE = "linear"
    NON_LINEAR_CORPUS_TYPE = "non_linear"

    CORPORA_DATA            = os.path.join(CORPORA, "extracted_data")
    LINEAR_CORPORA_DATA     = os.path.join(CORPORA_DATA, LINEAR_CORPUS_TYPE)
    NON_LINEAR_CORPORA_DATA = os.path.join(CORPORA_DATA, NON_LINEAR_CORPUS_TYPE)

    CORPORA_TEXTS            = os.path.join(CORPORA, "raw_texts")
    LINEAR_CORPORA_TEXTS     = os.path.join(CORPORA_TEXTS, LINEAR_CORPUS_TYPE)
    NON_LINEAR_CORPORA_TEXTS = os.path.join(CORPORA_TEXTS, NON_LINEAR_CORPUS_TYPE)

    # KEYED_VECTOR_FILE = os.path.join(DATA, "KeyedVectors_objects", "object0")
    KEYWORD_EXTRACTION_PATH = os.path.join(SRC, "keywords_extraction")
    KEYED_VECTOR_FILE = os.path.join(DATA, "KeyedVectors_objects", "object0")
    SUPERVISED_DATASET_PATH = os.path.join(DATA, "supervised_learning_dataset")
    WIKIPEDIA_ENTITIES_PATH = os.path.join(DATA, "wikipedia_entities")

    STUDENT_SIMUL = os.path.join(SRC, "student_simulation")
    DATASET_STATS = os.path.join(STUDENT_SIMUL, "dataset_stats")

    MODEL = os.path.join(SRC, "model")

    BASELINE_BASSEN = os.path.join(SRC, 'baseline_bassen')
    BASSEN_CONFIGS = os.path.join(BASELINE_BASSEN, 'configs')

    ANALYZE_RESULTS = os.path.join(SRC, 'analyze_results')
    STATS_RESULTS   = os.path.join(ANALYZE_RESULTS, 'stats')
    PLOTS_RESULTS   = os.path.join(ANALYZE_RESULTS, 'plots')


    def __init__(self):
        pass

    @classmethod
    def BROWSE_CORPORA(cls,
                       data_types:Union[str, List[str]]="all",
                       corpus_types:Union[str, List[str]]="all",
                       corpus_groups:Union[str, List[str]]="all",
                       corpus_names:Union[str, List[str]]="all",
                       without_corpus_groups:Union[str, List[str]]=[],
                       without_corpus_names:Union[str, List[str]]=[]):
        data_types = cls.GET_ALL_DATA_TYPES() if data_types=="all" else PathManager.to_list(data_types)
        for data_type in data_types:
            corpus_types = cls.GET_ALL_CORPUS_TYPES(data_type=data_type) if corpus_types=="all" else PathManager.to_list(corpus_types)
            for corpus_type in corpus_types:
                corpus_groups_list = cls.GET_ALL_CORPUS_GROUPS(data_type=data_type, corpus_type=corpus_type) if corpus_groups=="all" else PathManager.to_list(corpus_groups)
                corpus_groups_list = [item for item in corpus_groups_list if item not in without_corpus_groups]
                for corpus_group in corpus_groups_list:
                    corpus_names_list = cls.GET_ALL_CORPUS_NAMES(data_type=data_type, corpus_type=corpus_type, corpus_group=corpus_group) if corpus_names=="all" else PathManager.to_list(corpus_names)
                    corpus_names_list = [item for item in corpus_names_list if item not in without_corpus_names]
                    for corpus_name in corpus_names_list:
                        path = cls.GET_CORPUS_PATH(data_type, corpus_type, corpus_group, corpus_name)
                        if os.path.isdir(path):
                            yield corpus_name, corpus_group, corpus_type, data_type, path
                        else:
                            print(f'\n!! Rejected path: {path}\n')
    
    
    @classmethod
    def GET_CORPUS_PATH(cls, data_type, corpus_type, corpus_group, corpus_name):
        return os.path.join(cls.CORPORA, data_type, corpus_type, corpus_group, corpus_name)
    

    @classmethod
    def GET_ALL_DATA_TYPES(cls):
        path = cls.CORPORA
        all_data_types = utils.extract_folders_from_dir(path)
        return all_data_types

    @classmethod
    def GET_ALL_CORPUS_TYPES(cls, data_type):
        data_path = cls.DATA_TYPE2PATH(data_type)
        corpus_types = utils.extract_folders_from_dir(data_path)
        return corpus_types

    @classmethod
    def GET_ALL_CORPUS_GROUPS(cls, data_type, corpus_type):
        corpus_type_path = cls.CORPUS_TYPE2PATH(data_type, corpus_type)
        corpus_groups = utils.extract_folders_from_dir(corpus_type_path)
        return corpus_groups
    
    @classmethod
    def GET_ALL_CORPUS_NAMES(cls, data_type:str, corpus_type:str, corpus_group:str):
        path = cls.CORPUS_GROUP2PATH(data_type, corpus_type, corpus_group)
        corpus_names = utils.extract_folders_from_dir(path)
        return corpus_names
    

    @classmethod
    def DATA_TYPE2PATH(cls, data_type):
        if data_type == "extracted_data":
            return cls.CORPORA_DATA
        elif data_type == "raw_texts":
            return cls.CORPORA_TEXTS
        else:
            raise Exception(f'data_type "{data_type}" not supported.')
    
    @classmethod
    def CORPUS_TYPE2PATH(cls, data_type, corpus_type):
        data_type_path = cls.DATA_TYPE2PATH(data_type)
        path = os.path.join(data_type_path, corpus_type)
        return path
    
    @classmethod
    def CORPUS_GROUP2PATH(cls, data_type, corpus_type, corpus_group):
        corpus_type_path = cls.CORPUS_TYPE2PATH(data_type, corpus_type)
        path = os.path.join(corpus_type_path, corpus_group)
        return path
    

    @staticmethod
    def to_list(obj: Union[str, List[str]]):
        if isinstance(obj, str):
            return [obj]
        elif isinstance(obj, list):
            return obj
        else:
            raise Exception(f'Expected types "list" or "str" for object "{obj}".')