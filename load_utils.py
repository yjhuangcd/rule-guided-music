import importlib
import torch
from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model(name, ckpt):
    config = OmegaConf.load(f'taming-transformers/configs/pr/{name}.yaml')
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt) # load_state_dict(mc['state_dict'])
    model.eval()
    return model


def load_data(name):
    config = OmegaConf.load(f'taming-transformers/configs/pr/{name}.yaml')
    data = instantiate_from_config(config.data)
    return data
