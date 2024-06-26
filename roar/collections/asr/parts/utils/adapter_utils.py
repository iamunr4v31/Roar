from dataclasses import is_dataclass

import torch
from omegaconf import DictConfig, OmegaConf

from roar.utils import logging

# Constants
LINEAR_ADAPTER_CLASSPATH = "roar.collections.common.parts.adapter_modules.LinearAdapter"
MHA_ADAPTER_CLASSPATH = "roar.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.MultiHeadAttentionAdapter"  # TODO: Add these adapters
RELMHA_ADAPTER_CLASSPATH = "roar.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.RelPositionMultiHeadAttentionAdapter"
POS_ENCODING_ADAPTER_CLASSPATH = "roar.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.PositionalEncodingAdapter"
REL_POS_ENCODING_ADAPTER_CLASSPATH = "roar.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.RelPositionalEncodingAdapter"


def convert_adapter_cfg_to_dict_config(cfg: DictConfig):
    # Convert to DictConfig from dict or Dataclass
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not isinstance(cfg, DictConfig):
        cfg = DictConfig(cfg)

    return cfg


def update_adapter_cfg_input_dim(
    module: torch.nn.Module, cfg: DictConfig, *, module_dim: int
):
    """
    Update the input dimension of the provided adapter config with some default value.

    Args:
        module: The module that implements AdapterModuleMixin.
        cfg: A DictConfig or a Dataclass representing the adapter config.
        module_dim: A default module dimension, used if cfg has an incorrect input dimension.

    Returns:
        A DictConfig representing the adapter's config.
    """
    cfg = convert_adapter_cfg_to_dict_config(cfg)

    input_dim_valid_keys = ["in_features", "n_feat"]
    input_key = None

    for key in input_dim_valid_keys:
        if key in cfg:
            input_key = key
            break

    if input_key is None:
        raise ValueError(
            f"Failed to infer the input dimension of the Adapter cfg. \nExpected one of : {input_dim_valid_keys}.\n"
            f"Provided config : \n"
            f"{OmegaConf.to_yaml(cfg)}"
        )

    input_dim = cfg[input_key]

    if input_dim != module_dim:
        logging.info(
            f"Updating {module.__class__.__name__} Adapter input dim from {input_dim} to {module_dim}"
        )
        input_dim = module_dim

    cfg[input_key] = input_dim
    return cfg
