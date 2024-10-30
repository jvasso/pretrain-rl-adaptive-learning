import random
import numpy as np
import torch

from ..student_simulation import Student
from ..student_simulation.types import Feedback


def set_all_seeds(seed:int, device='cpu'):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # torch.backends.cudnn.enabled       = False
    elif str(device)=='cpu':
        pass
    else:
        raise ValueError(f'Device {device} not supported.')


def knows_all_target_kc(student:Student, target_kc_list:list):
    for kc in target_kc_list:
        if not student.knows_kc(kc=kc):
            return False
    return True


def is_terminated(student:Student, step_count:int, horizon:int, target_kc_list:list):
    if step_count == horizon - 1:
        return True, f"ep max length ({step_count})"
    elif knows_all_target_kc(student=student, target_kc_list=target_kc_list):
        return True, "knows all target kc"
    elif student.has_completed_corpus():
        return True, "corpus completed"
    else:
        return False, ""



