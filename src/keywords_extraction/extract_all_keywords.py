import os

from .classes.keywords_extractor import KeywordsExtractor
from ..path_manager import PathManager
from .. import utils


systems_path = "./src/keywords_extraction/prompts/templates/system_templates/"
humans_path  = "./src/keywords_extraction/prompts/templates/human_templates/"


### SELECT KEYWORDS EXTRACTOR

config = dict(llm_type="openai",
              model_name="gpt4",
              system_template=f"{systems_path}template7.txt",
              human_template=f"{humans_path}template1.txt",
              use_azure=True,
              temperature=0,
              max_tokens=1000)
keywords_extractor = KeywordsExtractor(**config)


### SELECT CORPORA

data_types    = "raw_texts"
corpus_types  = "linear"
corpus_groups = "new_corpora"
without_corpus_names = ["Machine_Learning", "Machine_Learning_with_Python", "Statistics", "Deep_Learning_Fundamentals_-_Intro_to_Neural_Networks"]


for corpus_name, corpus_group, corpus_type, data_type, corpus_text_path in PathManager.BROWSE_CORPORA(data_types=data_types,
                                                                                                      corpus_types=corpus_types,
                                                                                                      corpus_groups=corpus_groups,
                                                                                                      without_corpus_names=without_corpus_names):
            print(f'\n\n##############################################################')
            print(f'### NEW CORPUS: {corpus_name} ###')
            print(f'##############################################################')
            corpus_dict = keywords_extractor.extract_keywords_from_corpus(corpus_text_path=corpus_text_path,
                                                                          corpus_type=corpus_type,
                                                                          load=True,
                                                                          save=True,
                                                                          corpus_group=corpus_group,
                                                                          corpus_name=corpus_name)