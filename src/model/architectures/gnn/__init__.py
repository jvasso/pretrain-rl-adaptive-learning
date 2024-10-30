from typing import Dict, List
from . import hgt, transformer


def is_arch(name: str) -> bool:
    return name.startswith('arch') and name[4:].isdigit()


# Constructing the dictionary dynamically with modified key names
hgt_names_dict: Dict[str, List[Dict[str, str]]] = {
    f"hgt{name.split('arch')[1]}": getattr(hgt, name) for name in dir(hgt) if is_arch(name)
}
transformer_names_dict: Dict[str, List[Dict[str, str]]] = {
    f"transformer{name.split('arch')[1]}": getattr(transformer, name) for name in dir(transformer) if is_arch(name)
}

arch_names_dict = {**hgt_names_dict, **transformer_names_dict}