from typing import List, Dict
from . import linear


def is_arch(name: str) -> bool:
    return name.startswith('arch') and name[4:].isdigit()


# Constructing the dictionary dynamically with modified key names
arch_names_dict: Dict[str, List[Dict[str, str]]] = {
    f"linear{name[4:]}": getattr(linear, name) for name in dir(linear) if is_arch(name)
}