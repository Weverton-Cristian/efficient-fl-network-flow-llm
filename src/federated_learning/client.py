import os
import json
from collections import Counter
from dataclasses import dataclass
import inspect
from typing import Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel
from datasets import load_from_disk

from src.utils.hf import hf_from_pretrained_kwargs, hf_set_dtype_arg
from src.utils.peft_utils import prepare_peft_model_for_mixed_precision


@dataclass
class CausalLMDataCollatorWithPadding:
    """
    Dynamic padding + label masking for causal language modeling.

    We intentionally mask padded positions using attention_mask (not pad_token_id),
    because this project sets pad_token = eos_token for GPT-like tokenizers.
    """

    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class FedProxTrainer(Trainer):
    """
    Custom Trainer that implements FedProx by adding a proximal term to the loss.

    FedProx adds a regularization term that penalizes deviation from the global model:
    L_total = L_task + (mu/2) * ||w - w_global||^2

    This helps reduce client drift in non-IID scenarios.
    Reference: https://arxiv.org/abs/1812.06127
    """

    def __init__(self, global_state_dict=None, fedprox_mu=0.01, **kwargs):
        super().__init__(**kwargs)
        self.global_state_dict = global_state_dict
        self.fedprox_mu = fedprox_mu

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the total loss = task loss + proximal term.
        """
        # Compute standard task loss
        outputs = model(**inputs)
        task_loss = outputs.loss

        # Add proximal term if FedProx is enabled (mu > 0 and global weights available)
        if self.fedprox_mu > 0 and self.global_state_dict is not None:
            proximal_loss = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.global_state_dict:
                    global_param = self.global_state_dict[name].to(param.device)
                    proximal_loss += ((param - global_param) ** 2).sum()

            total_loss = task_loss + (self.fedprox_mu / 2.0) * proximal_loss
        else:
            total_loss = task_loss

        return (total_loss, outputs) if return_outputs else total_loss


class ClientTrainer:
    """
    Manages the training process for a single client in a federated learning round.
    """

    def __init__(self, client_id, config, gpu_id=0):
        self.client_id = client_id
        self.config = config
        self.gpu_id = gpu_id  # Armazena o ID da GPU
        self.model_name = config["model_name"]
        self.use_lora = config["lora"]
        self.training_task = str(
            config.get("training_task", "causal_lm")
        ).strip().lower()

    def _resolve_train_model_dtype(self):
        dtype_name = str(self.config.get("train_torch_dtype", "auto")).strip().lower()
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

    def _resolve_trainer_precision_flags(self):
        precision = str(
            self.config.get("train_mixed_precision", "auto")
        ).strip().lower()
        train_dtype = self._resolve_train_model_dtype()
        if not torch.cuda.is_available():
            return False, False

        if precision in ("none", "no", "false", "off"):
            return False, False
        if precision == "bf16":
            return False, True
        if precision == "fp16":
            return True, False
        if precision != "auto":
            raise ValueError(
                "Unsupported train_mixed_precision. Use auto, fp16, bf16 or none."
            )

        if train_dtype == torch.bfloat16:
            return False, True
        if train_dtype == torch.float16:
            return True, False
        return False, False

    def _prepare_peft_model_for_mixed_precision_training(self, model):
        target_dtype = self._resolve_train_model_dtype()
        prepare_peft_model_for_mixed_precision(model, target_dtype)

    def _validate_trainable_param_dtypes(self, model, *, use_fp16: bool, use_bf16: bool):
        trainable = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not trainable:
            raise RuntimeError(
                "No trainable parameters were found on the client model. "
                "This would make local training a no-op."
            )

        counts = Counter(str(param.dtype) for _, param in trainable)
        print(f"  Trainable dtype summary: {dict(counts)}")

        if use_fp16 or use_bf16:
            offenders = [
                (name, str(param.dtype))
                for name, param in trainable
                if param.dtype != torch.float32
            ]
            if offenders:
                sample = ", ".join(f"{name}={dtype}" for name, dtype in offenders[:5])
                mode = "fp16" if use_fp16 else "bf16"
                raise RuntimeError(
                    f"Invalid mixed-precision setup for {mode}: trainable PEFT parameters "
                    f"must be float32, but found {len(offenders)} offending tensors. "
                    f"Examples: {sample}"
                )

    def _load_model_for_training(self, round_num):
        """Loads the global model from the previous round and prepares it for training."""
        hf_kwargs = hf_from_pretrained_kwargs(self.config)
        train_dtype = self._resolve_train_model_dtype()
        hf_kwargs = hf_set_dtype_arg(hf_kwargs, train_dtype)
        model_path = os.path.join(
            self.config["results_path"],
            self.config["simulation_name"],
            f"round_{round_num - 1}",
            "global_model",
        )
        print(f"Client {self.client_id}: Loading model from {model_path}")

        legacy_4bit = bool(self.config.get("legacy_4bit_training", False))

        if self.training_task == "sequence_classification":
            if self.use_lora:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=int(self.config.get("num_labels", 2)),
                    **hf_kwargs,
                )
                if torch.cuda.is_available():
                    model = model.to(f"cuda:{self.gpu_id}")
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                self._prepare_peft_model_for_mixed_precision_training(model)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=int(self.config.get("num_labels", 2)),
                    **hf_kwargs,
                )
        elif "bert" in self.model_name.lower():
            if self.use_lora:
                model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name, **hf_kwargs
                )
                if torch.cuda.is_available():
                    model = model.to(f"cuda:{self.gpu_id}")
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                self._prepare_peft_model_for_mixed_precision_training(model)
            else:
                model = AutoModelForMaskedLM.from_pretrained(model_path, **hf_kwargs)
        else:
            if self.use_lora:
                extra_kwargs = {}
                if legacy_4bit:
                    # Match legado/utils.py quantization_config (NF4 4-bit).
                    extra_kwargs.update(
                        {
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_use_double_quant": True,
                        }
                    )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **extra_kwargs,
                    **hf_kwargs,
                )
                # In legacy 4-bit mode, avoid forcing .to() (bnb/accelerate handles placement).
                if torch.cuda.is_available() and not legacy_4bit:
                    model = model.to(f"cuda:{self.gpu_id}")
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                self._prepare_peft_model_for_mixed_precision_training(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)

        return model

    def train(self, round_num, learning_rate):
        """
        Executes a single round of training for the client.
        Supports FedProx regularization when fedprox_mu > 0 in config.
        """
        print(
            f"--- Starting training for Client {self.client_id} in Round {round_num} ---"
        )

        # 1. Load model and dataset
        model = self._load_model_for_training(round_num)
        if bool(self.config.get("legacy_flresults_layout", False)):
            client_dataset_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                "round_0",
                f"client_{self.client_id}",
            )
        else:
            client_dataset_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                "client_data",
                f"client_{self.client_id}",
            )
        client_dataset = load_from_disk(client_dataset_path)
        if len(client_dataset) == 0:
            print(f"  Client {self.client_id}: No local samples. Skipping training.")
            return None
        client_dataset = client_dataset.shuffle(seed=round_num)

        # 2. Capture global model state for FedProx (before training)
        fedprox_mu = self.config.get("fedprox_mu", 0.0)
        global_state_dict = None
        if fedprox_mu > 0:
            # Clone the global model parameters (only trainable ones for efficiency)
            global_state_dict = {
                name: param.data.clone().detach()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            print(f"  FedProx enabled with mu={fedprox_mu}")

        # 3. Setup Training Arguments
        legacy_training_args = bool(self.config.get("legacy_training_args", False))
        ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
        dataloader_num_workers = int(self.config.get("dataloader_num_workers", 0))
        dataloader_pin_memory = bool(self.config.get("dataloader_pin_memory", True))
        dataloader_persistent_workers = bool(
            self.config.get("dataloader_persistent_workers", False)
        )
        max_steps = int(self.config.get("max_steps", -1))
        num_train_epochs = float(self.config.get("num_train_epochs", 1.0))
        logging_steps = int(self.config.get("logging_steps", 1000))
        warmup_steps = self.config.get("warmup_steps")
        warmup_ratio = self.config.get("warmup_ratio")
        use_fp16, use_bf16 = self._resolve_trainer_precision_flags()
        self._validate_trainable_param_dtypes(
            model, use_fp16=use_fp16, use_bf16=use_bf16
        )

        def _maybe_set_dataloader_args(kwargs):
            if "dataloader_num_workers" in ta_params:
                kwargs["dataloader_num_workers"] = dataloader_num_workers
            if "dataloader_pin_memory" in ta_params:
                kwargs["dataloader_pin_memory"] = dataloader_pin_memory
            if "dataloader_persistent_workers" in ta_params:
                kwargs["dataloader_persistent_workers"] = (
                    dataloader_persistent_workers and dataloader_num_workers > 0
                )
            return kwargs

        def _maybe_set_warmup_args(kwargs):
            # Prefer warmup_steps because warmup_ratio is deprecated in the
            # transformers version used in this environment.
            if warmup_steps is not None and "warmup_steps" in ta_params:
                kwargs["warmup_steps"] = int(warmup_steps)
            elif warmup_ratio is not None and "warmup_ratio" in ta_params:
                kwargs["warmup_ratio"] = float(warmup_ratio)
            return kwargs

        if legacy_training_args:
            # transformers renamed evaluation_strategy -> eval_strategy (>=4.57)
            eval_key = (
                "eval_strategy"
                if "eval_strategy" in ta_params
                else "evaluation_strategy"
            )

            ta_kwargs = {
                "output_dir": "./fl-results",
                "logging_steps": logging_steps,
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
                "num_train_epochs": num_train_epochs,
                "save_steps": 1000,
                "fp16": use_fp16,
                "bf16": use_bf16,
                "optim": "paged_adamw_8bit",
                "per_device_train_batch_size": self.config["batch_size"],
                "gradient_accumulation_steps": self.config.get(
                    "gradient_accumulation_steps", 1
                ),
                "lr_scheduler_type": self.config.get(
                    "trainer_lr_scheduler_type", self.config["lr_scheduler_type"]
                ),
                "max_grad_norm": float(self.config.get("max_grad_norm", 1.0)),
                "save_strategy": "no",
            }
            if max_steps > 0:
                ta_kwargs["max_steps"] = max_steps
                ta_kwargs[eval_key] = "steps"
                ta_kwargs["eval_steps"] = max_steps + 1
            else:
                ta_kwargs["max_steps"] = -1
                ta_kwargs[eval_key] = "no"
            ta_kwargs = _maybe_set_warmup_args(ta_kwargs)
            ta_kwargs = _maybe_set_dataloader_args(ta_kwargs)
            training_args = TrainingArguments(**ta_kwargs)
        else:
            ta_kwargs = {
                "output_dir": os.path.join(
                    self.config["results_path"],
                    self.config["simulation_name"],
                    "client_training_output",
                ),
                "logging_steps": logging_steps,
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
                "fp16": use_fp16,
                "bf16": use_bf16,
                "optim": "paged_adamw_8bit",
                "per_device_train_batch_size": self.config["batch_size"],
                "gradient_accumulation_steps": self.config.get(
                    "gradient_accumulation_steps", 1
                ),
                "lr_scheduler_type": self.config.get(
                    "trainer_lr_scheduler_type", self.config["lr_scheduler_type"]
                ),
                "max_grad_norm": float(self.config.get("max_grad_norm", 1.0)),
                "save_strategy": "no",  # We save manually
                "num_train_epochs": num_train_epochs,
            }
            if max_steps > 0:
                ta_kwargs["max_steps"] = max_steps
            else:
                ta_kwargs["max_steps"] = -1
            ta_kwargs = _maybe_set_warmup_args(ta_kwargs)
            ta_kwargs = _maybe_set_dataloader_args(ta_kwargs)
            training_args = TrainingArguments(
                **ta_kwargs
            )

        legacy_eval_enabled = False
        if legacy_training_args:
            eval_attr = (
                getattr(training_args, "eval_strategy", None)
                if hasattr(training_args, "eval_strategy")
                else getattr(training_args, "evaluation_strategy", "no")
            )
            eval_mode = getattr(eval_attr, "value", eval_attr)
            legacy_eval_enabled = str(eval_mode) != "no"

        # 4. Setup Trainer (FedProxTrainer if mu > 0, else standard Trainer)
        use_legacy_trainer = self.config.get("use_legacy_trainer", False)
        tr_params = set(inspect.signature(Trainer.__init__).parameters)

        if self.training_task == "sequence_classification":
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hf_from_pretrained_kwargs(self.config)
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            if (
                getattr(model.config, "pad_token_id", None) is None
                and tokenizer.pad_token_id is not None
            ):
                model.config.pad_token_id = tokenizer.pad_token_id

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": client_dataset,
                "data_collator": data_collator,
            }
            if legacy_eval_enabled:
                trainer_kwargs["eval_dataset"] = client_dataset
            if use_legacy_trainer:
                if "processing_class" in tr_params:
                    trainer_kwargs["processing_class"] = tokenizer
                elif "tokenizer" in tr_params:
                    trainer_kwargs["tokenizer"] = tokenizer

            if fedprox_mu > 0:
                trainer = FedProxTrainer(
                    global_state_dict=global_state_dict,
                    fedprox_mu=fedprox_mu,
                    **trainer_kwargs,
                )
            else:
                trainer = Trainer(**trainer_kwargs)
        elif "bert" in self.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hf_from_pretrained_kwargs(self.config)
            )
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": client_dataset,
                "data_collator": data_collator,
            }
            if legacy_eval_enabled:
                trainer_kwargs["eval_dataset"] = client_dataset
            if fedprox_mu > 0:
                trainer = FedProxTrainer(
                    global_state_dict=global_state_dict,
                    fedprox_mu=fedprox_mu,
                    **trainer_kwargs,
                )
            else:
                trainer = Trainer(**trainer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hf_from_pretrained_kwargs(self.config)
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

            # Legacy mode: don't use data_collator (like original utils.py)
            if use_legacy_trainer:
                data_collator = None
            else:
                data_collator = CausalLMDataCollatorWithPadding(tokenizer=tokenizer)

            if fedprox_mu > 0:
                trainer_kwargs = {
                    "model": model,
                    "args": training_args,
                    "train_dataset": client_dataset,
                    "data_collator": data_collator,
                }
                if legacy_eval_enabled:
                    trainer_kwargs["eval_dataset"] = client_dataset
                if use_legacy_trainer:
                    if "processing_class" in tr_params:
                        trainer_kwargs["processing_class"] = tokenizer
                    elif "tokenizer" in tr_params:
                        trainer_kwargs["tokenizer"] = tokenizer

                trainer = FedProxTrainer(
                    global_state_dict=global_state_dict,
                    fedprox_mu=fedprox_mu,
                    **trainer_kwargs,
                )
            else:
                trainer_kwargs = {
                    "model": model,
                    "args": training_args,
                    "train_dataset": client_dataset,
                    "data_collator": data_collator,
                }
                if legacy_eval_enabled:
                    trainer_kwargs["eval_dataset"] = client_dataset
                if use_legacy_trainer:
                    if "processing_class" in tr_params:
                        trainer_kwargs["processing_class"] = tokenizer
                    elif "tokenizer" in tr_params:
                        trainer_kwargs["tokenizer"] = tokenizer

                trainer = Trainer(**trainer_kwargs)

        # 5. Run Training
        trainer.train()
        print(f"--- Client {self.client_id} training complete. ---")

        # 5. Extract LoRA adapters to CPU and return them
        # This prevents VRAM overflow on the server by not returning the whole model.
        if self.use_lora:
            cpu_adapters = {
                name: param.detach().to("cpu").clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
        else:
            # For full fine-tuning, return the entire state dict on CPU
            cpu_adapters = {
                name: param.to("cpu") for name, param in model.state_dict().items()
            }

        # Explicitly free up VRAM
        del model
        del trainer
        torch.cuda.empty_cache()

        return cpu_adapters
