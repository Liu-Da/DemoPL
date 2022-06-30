from typing import MappingView

from omegaconf import OmegaConf


class InstantiationException(Exception):
    ...

def _locate(path):
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj

def instantiate(config, *args, **kwargs):

    # Return None if config is None
    if config is None:
        return None

    if isinstance(config, MappingView): config = list(config)

    config = OmegaConf.structured(config, flags={"allow_objects": True})

    if OmegaConf.is_dict(config):
        if kwargs:
            config = OmegaConf.merge(config, kwargs)
        OmegaConf.resolve(config)

        if "target" not in config:
            return instantiate(config.values())
        else:
            return instantiate_node(config, *args)

    elif OmegaConf.is_list(config):
        OmegaConf.resolve(config)
        return instantiate_node(config, *args)
        
    else:
        msg = f"Cannot instantiate config of type {type(config).__name__}. Top level config must be an OmegaConf DictConfig/ListConfig object or a plain dict/list."
        raise InstantiationException(msg)

def instantiate_node(node, *args):

    # If OmegaConf list, create new list of instances if recursive
    if OmegaConf.is_list(node):
        items = [
            instantiate_node(item)
            for item in node._iter_ex(resolve=True)
        ]
        lst = OmegaConf.create(items, flags={"allow_objects": True})
        return lst  
    elif OmegaConf.is_dict(node):
        if "target" in node:
            target = _resolve_target(node.get("target"))
            kwargs = {}
            for key in node.keys():
                if key != "target":
                    kwargs[key] = node[key]
            return _call_target(target, args, kwargs)
        else:
            msg = f"There is no _target_ in config."
            raise InstantiationException(msg)
    else:
        msg = f"Unexpected config type : {type(node).__name__}."
        raise InstantiationException(msg)

def _call_target(target, args, kwargs):
    """Call target (type) with args and kwargs."""
    try:
        return target(*args, **kwargs)
    except Exception as e:
        msg = f"Error in call to target '{target}':\n{repr(e)}"
        raise InstantiationException(msg) from e

def _resolve_target(target):
    """Resolve target string into callable."""
    try:
        target = _locate(target)
    except Exception as e:
        msg = f"Error locating target '{target}', see chained exception above."
        raise InstantiationException(msg) from e
    return target