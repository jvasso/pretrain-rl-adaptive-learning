arch1 = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"}
]

arch2 = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"}
]

arch3 = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"}
]

arch4 = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"},
    {"layer_type": "doc"       , "layer_cls": "Linear"}
]

arch5 = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"},
    {"layer_type": "sub_docs"  , "layer_cls": "Sub"},
    {"layer_type": "doc"       , "layer_cls": "Linear"}
]

arch6 = [
    {"layer_type": "kw_doc"    , "layer_cls": "HeteroDictLinear"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw_doc"    , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"   , "layer_cls": "Mul"},
    {"layer_type": "doc_kw"    , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc", "layer_cls": "HGTConv"},
    {"layer_type": "sub_docs"  , "layer_cls": "Sub"}
]

arch7 = [
    {"layer_type": "doc"        , "layer_cls": "Linear"},
    {"layer_type": "kw"         , "layer_cls": "Linear"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv"}
]

arch8 = [
    {"layer_type": "doc"        , "layer_cls": "Linear"},
    {"layer_type": "kw"         , "layer_cls": "Linear"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv"},
    {"layer_type": "doc"        , "layer_cls": "Linear"},
]

arch9 = [
    {"layer_type": "doc"        , "layer_cls": "Linear"},
    {"layer_type": "doc"        , "layer_cls": "Linear"},
    {"layer_type": "kw"         , "layer_cls": "Linear"},
    {"layer_type": "kw"         , "layer_cls": "Linear"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv"},
    {"layer_type": "doc"        , "layer_cls": "Linear"},
]

arch10 = [
    {"layer_type": "doc"        , "layer_cls": "Linear"},
    {"layer_type": "kw"         , "layer_cls": "Linear"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2doc"     , "layer_cls": "HGTConv"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv"}
]

arch11 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv", 'doc_mode':"doc_embeddings"}
]

arch12 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv"}
]

arch13 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv"},
    {"layer_type": "sub_docs"   , "layer_cls": "Sub"}
]

arch14 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "HGTConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "HGTConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "HGTConv", 'doc_mode':"doc_embeddings"},
    {"layer_type": "sub_docs"   , "layer_cls": "Sub"}
]