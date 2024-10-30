# build all embeddings & one-hot encodings from a corpus
import os

from ..path_manager import PathManager
from .. import utils
from .data_processing_functions import embedding_functions


# keyedvector_file = PathManager.KEYED_VECTOR_FILE + "_1"
# embeddings_filename = os.path.join(PathManager.DATA, "global_word_embeddings_files", "enwiki_20180420_100d.txt")
# load_keyedvector    = False
keyedvector_file = PathManager.KEYED_VECTOR_FILE
embeddings_filename = None
load_keyedvector    = True

new_embedding_folder_name = input('Please choose a name for the new embeddings folder:')

# data_types    = ["extracted_data"]
# corpus_types  = ["non_linear"]
# corpus_groups = ["hand_made"]
# corpus_names  = ["intro_to_ml"]
without_corpus_groups = []
# without_corpus_names  = []
data_types    = "extracted_data"
corpus_types  = "linear"
corpus_groups = "new_corpora"
without_corpus_names = ["Machine_Learning", "Machine_Learning_with_Python"]

for corpus_name, corpus_group, corpus_type, data_type, corpus_dir in PathManager.BROWSE_CORPORA(data_types=data_types,
                                                                                                corpus_types=corpus_types,
                                                                                                corpus_groups=corpus_groups,
                                                                                                without_corpus_groups=without_corpus_groups,
                                                                                                without_corpus_names=without_corpus_names):
    print(corpus_name)
    kw_file_path   = os.path.join(corpus_dir, "doc2kw.json")
    raw_doc2kw_str = utils.load_json_file(file_path=kw_file_path)
    
    # build embeddings
    kw2embed, infos, rejected_keywords = embedding_functions.build_embeddings(kw_dict             = raw_doc2kw_str,
                                                                              embeddings_filename = embeddings_filename,
                                                                              load_keyedvector    = load_keyedvector,
                                                                              ask_not_found       = True,
                                                                              keyedvector_file    = keyedvector_file,
                                                                              verbose             = True)
    embeddings_folder_path = os.path.join(corpus_dir,"embedding")
    current_embedding_folder_path = os.path.join(embeddings_folder_path, new_embedding_folder_name)
    embedding_functions.save_embeddings(kw2embed,
                                        infos,
                                        data_dir=current_embedding_folder_path)
    
    if len(rejected_keywords) > 0:
        print(f'\nRejected keywords:\n{rejected_keywords}')
        print('\nWARNING: Update rejected keywords in "doc2kw.json" file?')
        update_rejected_keywords = input("Answer (y/n):")
        if update_rejected_keywords in ["y", "Y"]:
            update_rejected_keywords = input("Irreversible operation. Are you sure? (y/n)")
            if update_rejected_keywords in ["y", "Y"]:
                new_doc2kw_str = {}
                for doc, kw_list in raw_doc2kw_str.items():
                    new_doc2kw_str[doc] = [kw for kw in kw_list if kw not in rejected_keywords]
                utils.save_dict_as_json(new_doc2kw_str, file_path=kw_file_path)