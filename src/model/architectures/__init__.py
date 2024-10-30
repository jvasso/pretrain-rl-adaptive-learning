from . import gnn, feedback

GNN_ARCH_TYPE = "gnn"
FEEDBACK_ARCH_TYPE = "feedback"


def load_arch(arch_name:str, type:str):
    if type == GNN_ARCH_TYPE:
        return gnn.arch_names_dict[arch_name]
    elif type == FEEDBACK_ARCH_TYPE:
        return feedback.arch_names_dict[arch_name]
    else:
        raise ValueError(f'Architecture type "{type}" not supported.')
