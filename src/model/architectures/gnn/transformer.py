arch11 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv", 'doc_mode':"doc_embeddings"}
]

arch12 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"}
]

arch13 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"},
    {"layer_type": "sub_docs"   , "layer_cls": "Sub"}
]

arch14 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv", 'doc_mode':"doc_embeddings"},
    {"layer_type": "sub_docs"   , "layer_cls": "Sub"}
]

arch15 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"}
]

arch16 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"}
]

arch17 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"},
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"}
]

arch18 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"},
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"}
]

arch19 = [
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"}
]

arch20 = [
    {"layer_type": "doc_kw"     , "layer_cls": "Linear"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2doc"     , "layer_cls": "TransformerConv"},
    {"layer_type": "merge_f"    , "layer_cls": "Mul"},
    {"layer_type": "doc2kw"     , "layer_cls": "TransformerConv"},
    {"layer_type": "kw2all_doc" , "layer_cls": "TransformerConv"}
]