import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, TaskType, get_peft_model

from src.utils.hf import hf_from_pretrained_kwargs, hf_set_dtype_arg
from src.utils.peft_utils import prepare_peft_model_for_mixed_precision


def _get_training_task(config: dict) -> str:
    return str(config.get("training_task", "causal_lm")).strip().lower()


def _resolve_lora_target_modules(config: dict):
    target_modules = config.get("lora_target_modules", None)
    if isinstance(target_modules, str):
        return [m.strip() for m in target_modules.split(",") if m.strip()]
    return target_modules


def _resolve_modules_to_save(model, config: dict, training_task: str):
    configured = config.get("lora_modules_to_save")
    if isinstance(configured, str):
        configured = [m.strip() for m in configured.split(",") if m.strip()]
    if configured:
        return configured

    if training_task != "sequence_classification":
        return None

    candidate_names = {"score", "classifier", "classification_head"}
    resolved = []
    for module_name, _ in model.named_modules():
        leaf_name = module_name.split(".")[-1]
        if leaf_name in candidate_names and leaf_name not in resolved:
            resolved.append(leaf_name)
    return resolved or None


def _resolve_train_model_dtype(config: dict):
    dtype_name = str(config.get("train_torch_dtype", "auto")).strip().lower()
    if dtype_name in ("", "auto"):
        if torch.cuda.is_available():
            return torch.float16
        return None
    if dtype_name in ("float16", "fp16", "half"):
        return torch.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_name in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(
        "Unsupported train_torch_dtype. Use auto, float16, bfloat16 or float32."
    )


def _prepare_peft_model_for_mixed_precision_training(model, target_dtype) -> None:
    # PEFT recommends trainable parameters in FP32 when using mixed precision.
    # This keeps the frozen backbone in the requested reduced precision while
    # preventing GradScaler errors on LoRA / classifier-head gradients.
    #
    # We also install a dtype bridge for `modules_to_save` wrappers such as the
    # sequence-classification `score` head, whose FP32 weights otherwise receive
    # FP16 hidden states from the frozen backbone and fail at runtime.
    prepare_peft_model_for_mixed_precision(model, target_dtype)


def initialize_global_model(config):
    """
    Initializes a model and tokenizer from Hugging Face, applies LoRA if configured,
    and saves it as the initial global model for round 0.

    Args:
        config (dict): The experiment configuration dictionary.

    Returns:
        tuple: A tuple containing the initialized model and tokenizer.
    """
    model_name = config['model_name']
    use_lora = config['lora']
    training_task = _get_training_task(config)
    train_dtype = _resolve_train_model_dtype(config)

    print(f"Initializing model: {model_name}")
    hf_kwargs = hf_from_pretrained_kwargs(config)
    hf_kwargs = hf_set_dtype_arg(hf_kwargs, train_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)

    # Determine model type (e.g., BERT vs. GPT-like)
    if training_task == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=int(config.get("num_labels", 2)),
            **hf_kwargs,
        )
    else:
        if 'bert' in model_name.lower():
            model = AutoModelForMaskedLM.from_pretrained(model_name, **hf_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        print(f"Applying LoRA with rank={config['lora_rank']}...")
        target_modules = _resolve_lora_target_modules(config)
        modules_to_save = _resolve_modules_to_save(model, config, training_task)
        if target_modules:
            print(f"  LoRA target_modules: {target_modules}")
        if modules_to_save:
            print(f"  LoRA modules_to_save: {modules_to_save}")
        lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_rank'] * config['lora_alpha_multiplier'],
            lora_dropout=config['lora_dropout'],
            bias="none",
            task_type=TaskType.SEQ_CLS
            if training_task == "sequence_classification"
            else TaskType.CAUSAL_LM,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        _prepare_peft_model_for_mixed_precision_training(model, train_dtype)
        print("LoRA applied successfully.")
        model.print_trainable_parameters()

    # Save the initial model state for round 0
    initial_model_path = os.path.join(
        config['results_path'],
        config['simulation_name'],
        'round_0',
        'global_model'
    )
    os.makedirs(initial_model_path, exist_ok=True)
    model.save_pretrained(initial_model_path)

    print(f"Initial global model saved to: {initial_model_path}")

    return model, tokenizer
