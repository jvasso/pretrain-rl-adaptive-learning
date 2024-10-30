


def build_doc2kc_file(doc2kw_dict:dict, corpus_type:str):
    doc2kc_dict = {}
    if corpus_type == "linear":
        for doc in doc2kw_dict.keys():
            doc2kc_dict[doc] = [doc]
    elif corpus_type == "non_linear":
        for doc in doc2kw_dict.keys():
            doc_level, doc_sublevel = doc.split('_')
            if doc_sublevel == '0':
                kc_list = [f'{doc_level}_0',f'{doc_level}_1']
            elif doc_sublevel == '1':
                kc_list = [f'{doc_level}_1',f'{doc_level}_2']
            else:
                raise ValueError()
            doc2kc_dict[doc] = kc_list
    else:
        raise Exception(f"Corpus type {corpus_type} not supported.")
    return doc2kc_dict