import os

from ..path_manager import PathManager
from .. import utils
from .data_processing_functions import knowledge_components_functions

data_types   = ["extracted_data"]
# corpus_types = ["linear"]
# corpus_groups =["hand_made"]
# corpus_names  = ["corpus2","corpus3","corpus4","corpus5","corpus6"]
# corpus_types = ["non_linear"]
# corpus_groups =["hand_made"]
# corpus_names  = ["intro_to_ml"]
corpus_types  = "linear"
corpus_groups = "new_corpora"
without_corpus_names = ["Machine_Learning", "Machine_Learning_with_Python"]

for corpus_name, corpus_group, corpus_type, data_type, corpus_dir in PathManager.BROWSE_CORPORA(data_types=data_types,
                                                                                                corpus_types=corpus_types,
                                                                                                corpus_groups=corpus_groups,
                                                                                                without_corpus_names=without_corpus_names):
    kw_file_path = os.path.join(corpus_dir, "doc2kw.json")
    raw_doc2kw = utils.load_json_file(file_path=kw_file_path)
    doc2kc_dict = knowledge_components_functions.build_doc2kc_file(raw_doc2kw, corpus_type=corpus_type)
    
    file_path = os.path.join(corpus_dir, "doc2kc.json")
    utils.save_dict_as_json(data_dict=doc2kc_dict, file_path=file_path)
