import logging
from typing import Any, Dict, Union

import torch
import torch.nn
import torch.optim


def filter_state_dict(
    dst_state: Dict[str, Union[float, torch.Tensor]],
    src_state: Dict[str, Union[float, torch.Tensor]],
):
    """Filter name, size mismatch instances between dicts.

    Args:
        dst_state: reference state dict for filtering
        src_state: target state dict for filtering

    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            match_state[key] = value
        else:
            if key not in dst_state:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of name not found in target dict"
                )
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size mismatch"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
    return match_state


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>:<copy_Keys>
        revised by WingSing 2025-07-30
    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
        >>> load_pretrained_model("somewhere/model.pth::::separator.blocks", model)
    """
    sps = init_param.split(":", 5)
    if len(sps) == 5:
        path, src_key, dst_key, excludes, copy_keys = sps
    elif len(sps) == 4:
        path, src_key, dst_key, excludes = sps
        copy_keys = None
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes, copy_keys = None, None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes, copy_keys = None, None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes, copy_keys = None, None, None, None
    
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None
    if copy_keys == "":
        copy_keys = None
    if excludes == "":
        excludes = None

    if dst_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, dst_key)

    src_state = torch.load(path, map_location=map_location, weights_only=False)
    if "module" in src_state:
        src_state = src_state["module"]

    if excludes is not None:
        for e in excludes.split(","):
            src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    if src_key is not None:
        src_state = {
            k[len(src_key) + 1 :]: v
            for k, v in src_state.items()
            if k.startswith(src_key)
        }

    # Handle copy_keys functionality
    if copy_keys is not None:
        for copy_key in copy_keys.split(","):
            copy_key = copy_key.strip()
            if copy_key:
                _handle_copy_key(model, src_state, copy_key)

    dst_state = obj.state_dict()
    if ignore_init_mismatch:
        src_state = filter_state_dict(dst_state, src_state)
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)


def _handle_copy_key(model: torch.nn.Module, src_state: Dict[str, Union[float, torch.Tensor]], copy_key: str):
    """Handle copying parameters for ModuleList when current model has more modules.
    
    Args:
        model: The target model
        src_state: Source state dict from pretrained model
        copy_key: The key path to the ModuleList to copy (e.g., 'encoder.layers')
    """
    def get_attr(obj: Any, key: str):
        """Get an nested attribute."""
        if key.strip() == "":
            return obj
        for k in key.split("."):
            obj = getattr(obj, k)
        return obj
    
    try:
        # Get the target ModuleList
        target_module = get_attr(model, copy_key)
        
        # Check if it's a ModuleList
        if not isinstance(target_module, torch.nn.ModuleList):
            logging.warning(f"Copy key {copy_key} does not point to a ModuleList, skipping")
            return
        
        # Find source parameters for this copy_key
        src_params = {}
        prefix = copy_key + "."
        for key, value in src_state.items():
            # logging.info(f"prefix: {prefix}, key: {key}")
            if key.startswith(prefix):
                src_params[key] = value
        
        if not src_params:
            logging.warning(f"No source parameters found for copy key {copy_key}, skipping")
            return
        
        # Determine the number of modules in source
        src_module_indices = set()
        for key in src_params.keys():
            # Extract the module index (e.g., from "encoder.layers.0.weight" get 0)
            remaining_key = key[len(prefix):]
            if "." in remaining_key:
                try:
                    idx = int(remaining_key.split(".")[0])
                    src_module_indices.add(idx)
                except ValueError:
                    continue
        
        if not src_module_indices:
            logging.warning(f"No valid module indices found in source for copy key {copy_key}, skipping")
            return
        
        src_num_modules = max(src_module_indices) + 1
        target_num_modules = len(target_module)
        
        # Check if target has more modules than source
        if target_num_modules <= src_num_modules:
            logging.info(f"Target ModuleList {copy_key} has {target_num_modules} modules, "
                        f"source has {src_num_modules}, no copying needed")
            return
        
        # Check if all submodules in target are the same structure
        if not _are_submodules_same(target_module):
            logging.warning(f"Submodules in {copy_key} are not identical, skipping copy")
            return
        
        logging.info(f"Copying parameters from {src_num_modules} source modules to "
                    f"{target_num_modules} target modules for {copy_key}")
        
        # Copy parameters cyclically
        for target_idx in range(target_num_modules):
            src_idx = target_idx % src_num_modules
            
            # Copy all parameters from src_idx to target_idx
            for src_key, src_value in src_params.items():
                if src_key.startswith(f"{prefix}{src_idx}."):
                    # Create corresponding target key
                    suffix = src_key[len(f"{prefix}{src_idx}."):]
                    target_key = f"{prefix}{target_idx}.{suffix}"
                    src_state[target_key] = src_value
                    
    except Exception as e:
        logging.warning(f"Error handling copy key {copy_key}: {e}")


def _are_submodules_same(module_list: torch.nn.ModuleList) -> bool:
    """Check if all submodules in a ModuleList have the same structure.
    
    Args:
        module_list: The ModuleList to check
        
    Returns:
        True if all submodules have the same structure, False otherwise
    """
    if len(module_list) <= 1:
        return True
    
    # Get the state dict structure of the first module
    first_module_keys = set(module_list[0].state_dict().keys())
    first_module_shapes = {k: v.shape for k, v in module_list[0].state_dict().items()}
    
    # Check if all other modules have the same structure
    for i in range(1, len(module_list)):
        module_keys = set(module_list[i].state_dict().keys())
        module_shapes = {k: v.shape for k, v in module_list[i].state_dict().items()}
        
        if module_keys != first_module_keys or module_shapes != first_module_shapes:
            return False
    
    return True
