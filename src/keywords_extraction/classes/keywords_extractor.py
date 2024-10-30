from genericpath import isfile
import os
import pprint

from typing import List
import string

import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from os import path

import openai

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.llms import HumanInputLLM
from langchain.prompts import PromptTemplate

from .custom_output_parser import CustomOutputParser

from ... import utils
from ...student_simulation import CorpusGraph
from ...path_manager import PathManager


class KeywordsExtractor:
    
    def __init__(self,
                 template:str=None,
                 system_template:str=None,
                 human_template:str=None,
                 llm_type:str=None,
                 model_path:str=None,
                 model_name:str=None,
                 llm_params:dict={},
                 openai_api_key:str=None,
                 use_azure:bool=False,
                 temperature:int=0,
                 max_tokens:int=1000,
                 verbose:int=2):
        self.llm_type2prompt_func = {"human":self.build_chat_prompt, "openai":self.build_chat_prompt, "gpt4all":self.build_standard_prompt}
        assert (llm_type in self.llm_type2prompt_func.keys())

        self.template_file        = template
        self.system_template_file = system_template
        self.human_template_file  = human_template
        self.llm_type             = llm_type
        self.model_path           = model_path
        self.model_name           = model_name
        self.llm_params           = llm_params
        self.openai_api_key       = openai_api_key
        self.use_azure            = use_azure
        self.temperature          = temperature
        self.max_tokens           = max_tokens
        self.verbose              = verbose

        self._hyperparams = {"system_template_file":self.system_template_file.split("system_templates/")[1],
                             "human_template_file":self.human_template_file.split("human_templates/")[1],
                             "llm_type":self.llm_type,
                             "model_name":self.model_name}
        
        self._initialize_prompt()
        self._initialize_llm()
        
        self.output_parser = CustomOutputParser()
        self.llm_chain = self.prompt | self.llm | self.output_parser
    

    def _initialize_prompt(self):
        assert self.llm_type in self.llm_type2prompt_func, f"LLM type {self.llm_type} not supported"
        prompt_func = self.llm_type2prompt_func[self.llm_type]
        self.prompt = prompt_func()
    

    def _initialize_llm(self):
        if self.llm_type=="human":
            self.llm = HumanInputLLM()
        elif self.llm_type=="gpt4all" and path.exists(self.model_path):
            raise NotImplementedError()
        elif self.llm_type=="openai":
            if self.use_azure:
                self.set_openai_azure()
                self.llm = AzureChatOpenAI(deployment_name=self.model_name,
                                           max_tokens=self.max_tokens,
                                           temperature=self.temperature,
                                           openai_api_version=openai.api_version,
                                           openai_api_type=openai.api_type,
                                           openai_api_base=openai.api_base,
                                           openai_api_key=openai.api_key)
            else:
                assert (self.openai_api_key is not None) and (self.model_name is not None)
                assert self.model_name in {"gpt-4", "gpt-4-32k", "gpt-3.5", "gpt-3.5-turbo-1106","gpt-3.5-turbo"}
                self.llm = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)
        else:
            raise NotImplementedError()
    

    @staticmethod
    def set_openai_azure():
        os.environ['OPENAI_API_KEY'] = ''.join([line for line in open(os.path.join('src', 'keywords_extraction', 'openai_key.txt'))])
        openai.api_version = "2023-07-01-preview"
        openai.api_key = str(os.getenv("OPENAI_API_KEY"))
        openai.api_base = "https://ukmodels.openai.azure.com/"
        openai.api_type = "azure"
        assert openai.api_key != "None"


    def build_standard_prompt(self):
        assert (self.template_file is not None)
        template = utils.load_text_file(self.template_file)
        prompt = PromptTemplate(template=template, input_variables=["document"])
        return prompt

    def build_chat_prompt(self):
        assert (self.system_template_file is not None) and (self.human_template_file is not None)
        system_template = utils.load_text_file(self.system_template_file)
        human_template  = utils.load_text_file(self.human_template_file)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        return prompt
    

    def extract_keywords_from_text(self, text:str):
        keywords_list = self.llm_chain.invoke({"document":text})
        return keywords_list


    def evaluate_performance(self, save_predictions=True) -> dict:
        eval_dir = PathManager.EVAL_CORPORA
        perfs_per_doc = []
        
        for corpus_name in utils.extract_folders_from_dir(eval_dir):
            corpus_dict = {}
            raw_expected_results = utils.load_json_file(os.path.join(eval_dir,corpus_name,"expectations.json"))
            expected_results = self.preprocess_expected_results(raw_expected_results)

            for filename, content in utils.extract_files_from_dir(os.path.join(eval_dir,corpus_name)):
                doc_id = utils.find_single_number_in_string(filename)
                if doc_id in expected_results.keys():
                    if self.verbose >= 2: print(f"\n##### Doc ID: {doc_id} #####\n")
                    keywords_list = self.extract_keywords_from_text(content)
                    corpus_dict[doc_id] = keywords_list

            assert corpus_dict.keys() == expected_results.keys()

            if save_predictions:
                self.save_eval_predictions(corpus_dict)

            for doc_id, predicted_keywords in corpus_dict.items():
                true_keywords = expected_results[doc_id]
                results_dict = KeywordsExtractor.measure_perf_per_document(predicted_keywords=predicted_keywords, ground_truth_keywords=true_keywords)
                perfs_per_doc.append(results_dict.copy())
        
        global_perfs = utils.list_of_dicts_to_dict_of_lists(perfs_per_doc)
        global_results = {key:{"mean":np.mean(results), "std":np.std(results), "num_values":len(results)} for key,results in global_perfs.items()}
        
        return global_results # {'precision':{'mean':0.5, 'std':0.1, 'num_values':5}, 'recall':{...}, etc.}


    def save_eval_predictions(self, corpus_dict:dict):
        datetime = utils.current_datetime()
        file_path = os.path.join(PathManager.KEYWORD_EXTRACTION_PATH, "eval_predictions", f'{datetime}.json')
        utils.save_dict_as_json(data_dict=corpus_dict, file_path=file_path)



    def is_corpus_group(self):
        raise NotImplementedError()
    

    def is_corpus_name(self):
        raise NotImplementedError()
    

    def extract_keywords_from_corpus(self, corpus_text_path:str, corpus_type=str, load=True, save=True, corpus_group:str=None, corpus_name:str=None) -> dict:
        corpus_dict = {}
        infos = {}

        if load:
            assert (corpus_group is not None) and (corpus_name is not None)
            corpus_extracted_data_path = PathManager.GET_CORPUS_PATH(data_type="extracted_data",
                                                                     corpus_type=corpus_type, corpus_group=corpus_group, corpus_name=corpus_name)
            doc2kw_file_path = os.path.join(corpus_extracted_data_path, CorpusGraph.DOC2KW_FILE)
            corpus_dict      = utils.load_json_file(doc2kw_file_path, accept_none=True)
            if corpus_dict is None: corpus_dict = {}
        else:
            corpus_dict = {}
        
        for filename, content in utils.extract_files_from_dir(corpus_text_path, sort=True):
            if corpus_type=='non_linear':
                doc_id = filename.split('.')[0]
            else:
                doc_id = utils.find_single_number_in_string(filename)

            already_done = KeywordsExtractor.check_extracted_keywords_consistency(corpus_dict, doc_id)
            if already_done:
                pass
            else:
                keywords_list = self.extract_keywords_from_text(content)
                corpus_dict[doc_id] = keywords_list
                if save:
                    utils.save_dict_as_json(corpus_dict, doc2kw_file_path, create_folders=True)

        return corpus_dict

    @staticmethod
    def check_extracted_keywords_consistency(corpus_dict:dict, doc_id:str):
        if not doc_id in corpus_dict.keys():
            return False
        if not isinstance(corpus_dict[doc_id], list):
            return False
        if len(corpus_dict[doc_id])==0:
            return False
        return True
    

    @staticmethod
    def keys_with_identical_values(input_dict):
        value_to_keys = {}
        # Group keys with identical values
        for key, value in input_dict.items():
            value_to_keys.setdefault(value, []).append(key)
        # Filter out groups with only one key
        return [key_group for key_group in value_to_keys.values() if len(key_group) > 1]
    

    def preprocess_expected_results(self, expected_results):
        preprocessed_results = { key:expected_result for key,expected_result in expected_results.items() if len(expected_result)>0 }
        return preprocessed_results


    @staticmethod
    def compare_performance(extractors:List['KeywordsExtractor'], comparison_metric:str="f1_score", save_perfs=True):
        models = []
        means = []
        errors = []
        distinctive_hyperparams = list(KeywordsExtractor.find_distinct_features(extractors))
        if save_perfs:
            current_date = utils.current_datetime()
            perfs_path = os.path.join(PathManager.KEYWORD_EXTRACTION_PATH, 'eval_performance', current_date)
            os.mkdir(path=perfs_path)
        for extractor in extractors:
            print(f'\nNew extractor:\n{extractor}')
            global_perfs = extractor.evaluate_performance()
            if save_perfs:
                KeywordsExtractor.save_extractor_performance(perfs=global_perfs, extractor=extractor, perfs_path=perfs_path)
            result = global_perfs[comparison_metric]
            label_list = [ f"{key}:{extractor.hyperparams[key]}" for key in distinctive_hyperparams ]
            label_str = '\n'.join(label_list)
            models.append(label_str)
            means.append(result["mean"])
            errors.append(1.96 * (result["std"]/np.sqrt(result["num_values"]))) # Calculate the margin of error at 95% confidence interval, for normally distributed data, which is 1.96 * (std/sqrt(n))
        
        # Create a bar chart with error bars
        plt.bar(models, means, yerr=errors, capsize=5)
        plt.ylabel(comparison_metric)
        plt.title('Performance comparison of different prompts for keyword extraction')
        if save_perfs:
            figure_path = os.path.join(perfs_path, f'compare_performance-{utils.current_datetime()}.png')
            plt.savefig(figure_path, bbox_inches='tight')
        else:
            plt.show()
    

    @staticmethod
    def save_extractor_performance(perfs:dict, extractor:'KeywordsExtractor', perfs_path:str):
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
        keyword_extractor_path = os.path.join(perfs_path, f'{extractor.model_name}_{extractor.system_template_file.split("/")[-1].split(".")[0]}_{random_str}')
        os.mkdir(path=keyword_extractor_path)
        
        perfs_file_path = os.path.join(keyword_extractor_path, 'perfs.json')
        utils.save_dict_as_json(data_dict=perfs, file_path=perfs_file_path)
        
        hyperparams_file_path = os.path.join(keyword_extractor_path, 'hyperparams.json')
        utils.save_dict_as_json(data_dict=extractor._hyperparams, file_path=hyperparams_file_path)
    


    @staticmethod
    def find_distinct_features(extractors:List['KeywordsExtractor']):
        list_of_hyperparams = [extractor.hyperparams for extractor in extractors]
        unique_hyperparams = utils.extract_unique_keys(list_of_hyperparams)
        return unique_hyperparams

    
    @staticmethod
    def measure_perf(true_keywords, predicted_keywords):
        # Convert keywords to binary vectors
        all_possible_keywords = sorted(set(true_keywords + predicted_keywords))
        true_vector      = [int(keyword in true_keywords)      for keyword in all_possible_keywords]
        predicted_vector = [int(keyword in predicted_keywords) for keyword in all_possible_keywords]
        
        # Calculate metrics
        precision = precision_score(true_vector, predicted_vector) # proportion of predicted keywords that are correct
        recall = recall_score(true_vector, predicted_vector) # proportion of true keywords that were predicted
        f1 = f1_score(true_vector, predicted_vector) # harmonic mean of precision and recall, providing a single score that balances both
        jaccard = jaccard_score(true_vector, predicted_vector) # measures the similarity between the set of predicted keywords and the set of true keywords
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_similarity': jaccard
        }
    
    @staticmethod
    def measure_perf_per_document(predicted_keywords, ground_truth_keywords):
        true_positives = len(set(predicted_keywords) & set(ground_truth_keywords))
        precision = true_positives / len(predicted_keywords) if predicted_keywords else 0
        recall = true_positives / len(ground_truth_keywords) if ground_truth_keywords else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
        jaccard = true_positives / len(set(predicted_keywords) | set(ground_truth_keywords)) if predicted_keywords or ground_truth_keywords else 0
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_similarity':jaccard
        }
    
    
    

    @property
    def hyperparams(self):
        return self._hyperparams
    
    
    def __repr__(self) -> str:
        return self.to_text()
    def __str__(self) -> str:
        return self.to_text()
    
    def to_text(self):
        # params_dict = dict(
        #     system_template = self.system_template_file,
        #     human_template  = self.human_template_file,
        #     llm_type        = self.llm_type,
        #     model_name      = self.model_name,
        #     llm_params      = self.llm_params,
        #     temperature     = self.temperature,
        #     max_tokens      = self.max_tokens
        # )
        return pprint.pformat(self._hyperparams)

