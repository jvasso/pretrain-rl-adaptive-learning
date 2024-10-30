import os
import json
from datetime import datetime
import pprint

import numpy as np
from scipy import stats
import pandas as pd

from ..path_manager import PathManager
from . import utils



def preprocess_df(df:pd.DataFrame, method:str) -> pd.DataFrame:
    if method=='irt':
        df = df.drop(columns=[col for col in df.columns if 'step' in col])
    if 'episode' in df.columns:
        df = df.rename(columns={'episode': 'student'})
    else:
        if method=='irt':
            assert len(df)==10
            df['student'] = (df.index+1) * 5
        else:
            df['student'] = df.index * 5
    if 'Step' in df.columns:
        df = df.drop(columns=['Step'])
    df = df.drop(columns=[col for col in df.columns if 'MIN' in col or 'MAX' in col])

    assert 11 <= len(df.columns) <= 31
    return df


def mean_statistic(data, axis):
    return np.mean(data, axis=axis)


def compute_stats_student_bootstrap(data, confidence_level=0.95, n_resamples=1000):
    assert len(data.shape)==1
    data = data.reshape(-1, 1)
    res = stats.bootstrap(
            (data,),
            statistic=mean_statistic,
            vectorized=True,
            paired=False,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='percentile',  # 'BCa' can also be used for bias-corrected and accelerated
            random_state=42  # For reproducibility
    )
    lower_bound, upper_bound = res.confidence_interval
    original_mean = np.mean(data)
    assert len(upper_bound)==1 and len(lower_bound)==1
    return original_mean, upper_bound[0], lower_bound[0], None


def compute_stats_mean_and_std(data):
    mean = np.mean(data)
    std  = np.std(data)
    return mean, mean + std, mean - std, std


if __name__=='__main__':

    stat = 'mean'
    ci   = 'bootstrap' # std, bootstrap

    confidence_level = 0.95  if ci == 'bootstrap' else ''
    n_resamples      = 10000 if ci == 'bootstrap' else ''
    ci_infos = ci + '_' + str(confidence_level) + '_' + str(n_resamples)

    filename = datetime.now().strftime("%m-%d_%H-%M-%S") + '__' + stat + "__" + ci_infos + ".json"
    filepath = os.path.join(PathManager.STATS_RESULTS, filename)
    results_dict = {} # prior_knowledge => method => student step
    prior_knowledge_folder_path = os.path.join(PathManager.ANALYZE_RESULTS, 'results_target_task')
    prior_knowledge_list = utils.get_subfolders(folder_path=prior_knowledge_folder_path)

    for prior_knowledge in prior_knowledge_list:
        
        prior_knowledge_name = prior_knowledge
        results_dict[prior_knowledge_name] = {}

        methods_folder_path = os.path.join(prior_knowledge_folder_path, prior_knowledge)
        methods_list = utils.get_subfolders(folder_path=methods_folder_path)

        for method in methods_list:
            
            method_name = method
            results_dict[prior_knowledge_name][method_name] = {}

            result_folder_path = os.path.join(methods_folder_path, method)
            df = utils.load_csv_in_folder(folder_path=result_folder_path)
            df = preprocess_df(df, method)
            df.set_index('student', inplace=True)
            assert 10 <= len(df.columns) <= 30
            for step in df.index:
                data = df.loc[step].values
                N = len(data)
                assert N >= 10

                if stat=='mean':
                    if ci=='std':
                        central, upper_bound, lower_bound, std = compute_stats_mean_and_std(data)
                    elif ci=='bootstrap':
                        central, upper_bound, lower_bound, std = compute_stats_student_bootstrap(data, confidence_level=confidence_level, n_resamples=n_resamples)
                else:
                    raise ValueError(f'Not implemented.')
                
                results_dict[prior_knowledge_name][method_name][step] = dict(central=central, upper_bound=upper_bound, lower_bound=lower_bound)
                if std is not None:
                    results_dict[prior_knowledge_name][method_name][step]['std']=std
    
    pprint.pprint(results_dict)

    with open(filepath, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    
    print(f'Saved at: {filepath}')
                



            
