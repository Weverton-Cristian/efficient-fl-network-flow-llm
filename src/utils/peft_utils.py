import torch
from peft import cast_mixed_precision_params


def _resolve_weight_dtype(module) -> torch.dtype | None:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype

    for param in module.parameters(recurse=False):
        return param.dtype
    return None


def _resolve_modules_to_save_target(module):
    active_adapters = getattr(module, "active_adapters", None)
    modules_to_save = getattr(module, "modules_to_save", None)

    if active_adapters and modules_to_save:
        active_name = active_adapters[0]
        if active_name in modules_to_save:
            return modules_to_save[active_name]

    return getattr(module, "original_module", None)


def _cast_input_to_wrapped_weight_dtype(module, args, kwargs):
    if not args:
        return None

    x = args[0]
    if not isinstance(x, torch.Tensor):
        return None

    target_module = _resolve_modules_to_save_target(module)
    if target_module is None:
        return None

    target_dtype = _resolve_weight_dtype(target_module)
    if target_dtype is None or x.dtype == target_dtype:
        return None

    return (x.to(target_dtype), *args[1:]), kwargs


def install_modules_to_save_dtype_bridges(model) -> None:
    """
    Installs forward pre-hooks on PEFT `ModulesToSaveWrapper` instances so the
    wrapped module receives inputs in the same dtype as its own weights.

    This is particularly important for sequence-classification heads such as
    `score`, which remain trainable in float32 while the frozen backbone runs in
    float16/bfloat16 during mixed-precision training.
    """
    for module in model.modules():
        if not hasattr(module, "modules_to_save"):
            continue
        if getattr(module, "_flllm_dtype_bridge_installed", False):
            continue

        module.register_forward_pre_hook(
            _cast_input_to_wrapped_weight_dtype,
            with_kwargs=True,
        )
        module._flllm_dtype_bridge_installed = True


def prepare_peft_model_for_mixed_precision(model, target_dtype) -> None:
    """
    Applies PEFT mixed-precision casting and installs dtype bridges for wrapped
    trainable modules such as classification heads saved via `modules_to_save`.
    """
    if target_dtype is not None:
        cast_mixed_precision_params(model, dtype=target_dtype)
    install_modules_to_save_dtype_bridges(model)
