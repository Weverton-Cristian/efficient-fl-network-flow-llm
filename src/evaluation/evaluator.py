import os
from contextlib import nullcontext
import pandas as pd
import numpy as np
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

from src.utils.hf import hf_from_pretrained_kwargs, hf_set_dtype_arg
from src.utils.peft_utils import install_modules_to_save_dtype_bridges


class Evaluator:
    """
    Handles the evaluation of the trained global models from each round.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join(
            config["results_path"], config["simulation_name"]
        )
        self.output_dir = self.config.get("evaluation_output_dir", self.results_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.test_df = self._load_test_data()

    def _get_training_task(self) -> str:
        return str(self.config.get("training_task", "causal_lm")).strip().lower()

    def _is_sequence_classification(self) -> bool:
        return self._get_training_task() == "sequence_classification"

    def _anomaly_label_id(self) -> int:
        return int(self.config.get("anomaly_label_id", 1))

    def _score_descending_is_anomaly(self) -> bool:
        return self._is_sequence_classification()

    def _predict_from_scores(self, scores, threshold):
        scores = np.asarray(scores, dtype=float)
        if self._score_descending_is_anomaly():
            return (scores >= float(threshold)).astype(int)
        return (scores < float(threshold)).astype(int)

    def _build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], **hf_from_pretrained_kwargs(self.config)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return tokenizer

    def _load_model_from_path(self, model_path: str):
        eval_model_kwargs = self._eval_model_load_kwargs()
        if self._is_sequence_classification():
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_name"],
                num_labels=int(self.config.get("num_labels", 2)),
                **eval_model_kwargs,
            )
            if self.config["lora"]:
                model = PeftModel.from_pretrained(base_model, model_path)
                install_modules_to_save_dtype_bridges(model)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=int(self.config.get("num_labels", 2)),
                    **eval_model_kwargs,
                )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"], **eval_model_kwargs
            )
            if self.config["lora"]:
                model = PeftModel.from_pretrained(base_model, model_path)
                install_modules_to_save_dtype_bridges(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, **eval_model_kwargs
                )

        if (
            getattr(model.config, "pad_token_id", None) is None
            and getattr(self, "_cached_tokenizer_pad_id", None) is not None
        ):
            model.config.pad_token_id = self._cached_tokenizer_pad_id
        return model

    def _output_csv_path(self, basename: str) -> str:
        """
        Returns an output path for a CSV basename, suffixing by threshold mode
        when not using the oracle `f1_max` selection.
        """
        threshold_mode = self._get_threshold_mode()
        suffix = "" if threshold_mode == "f1_max" else f"_{threshold_mode}"
        return os.path.join(self.output_dir, f"{basename}{suffix}.csv")

    def _get_threshold_mode(self) -> str:
        """
        Resolves the configured threshold-selection strategy.

        Preferred key:
          - threshold_selection

        Backward-compatible alias:
          - threshold_strategy
        """
        return str(
            self.config.get(
                "threshold_selection",
                self.config.get("threshold_strategy", "f1_max"),
            )
        ).lower()

    def _load_test_data(self):
        """Loads and prepares the test dataset."""
        test_path = os.path.join(
            self.config["data_base_path"],
            self.config["dataset_name"],
            "processed",
            "test.csv",
        )
        df = pd.read_csv(test_path)
        # Optional: Balance the test set for more stable metrics
        # df = pd.concat([
        #     df[df['Label'] == 1].head(1000),
        #     df[df['Label'] == 0].head(1000)
        # ]).sample(frac=1).reset_index(drop=True)
        return df

    def _parse_timestamps(self, series: pd.Series) -> pd.Series:
        """
        Parses dataset timestamps robustly.

        WiFi/CICIDS2018 typically uses 24h timestamps like '%d/%m/%Y %H:%M:%S',
        while some older datasets in this project use 12h timestamps with AM/PM.
        We try both explicitly, then fall back to a permissive parser.
        """
        ts = pd.to_datetime(series, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        missing = ts.isna()
        if missing.any():
            ts.loc[missing] = pd.to_datetime(
                series[missing], format="%d/%m/%Y %I:%M:%S %p", errors="coerce"
            )
        missing = ts.isna()
        if missing.any():
            ts.loc[missing] = pd.to_datetime(
                series[missing], errors="coerce", dayfirst=True
            )
        return ts

    def _load_calibration_data(self):
        """
        Loads a benign-only calibration set for operational threshold selection.

        calibration_source:
          - "train_benign": uses processed/train.csv filtered to Label==0
          - "calibration_benign": uses processed/calibration.csv filtered to Label==0 (recommended)
          - "test_benign":  uses processed/test.csv filtered to Label==0 (not recommended for strict evaluation)
        """
        source = self.config.get("calibration_source", "train_benign")
        if source == "train_benign":
            split = "train"
        elif source == "calibration_benign":
            split = "calibration"
        else:
            split = "test"
        path = os.path.join(
            self.config["data_base_path"],
            self.config["dataset_name"],
            "processed",
            f"{split}.csv",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibration source '{source}' requires {path}, but it does not exist. "
                "Re-run preprocessing with calibration enabled in the WiFi split config."
            )
        df = pd.read_csv(path)
        if "Label" in df.columns:
            df = df[df["Label"] == 0].copy()

        n = int(self.config.get("calibration_num_samples", 2000))
        if n > 0 and len(df) > n:
            df = df.sample(n=n, random_state=0).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        return df

    def _is_in_top_k(self, top_k_preds, target_token):
        """Helper function: checks if target_token is in the top-k predictions."""
        return target_token in top_k_preds

    def _eval_model_load_kwargs(self):
        kwargs = hf_from_pretrained_kwargs(self.config)
        dtype_name = str(self.config.get("eval_torch_dtype", "")).lower().strip()
        if torch.cuda.is_available():
            if dtype_name in ("float16", "fp16", "half"):
                kwargs = hf_set_dtype_arg(kwargs, torch.float16)
            elif dtype_name in ("bfloat16", "bf16"):
                kwargs = hf_set_dtype_arg(kwargs, torch.bfloat16)
        return kwargs

    def _calculate_top_k_accuracy_for_texts(
        self, model, tokenizer, texts, *, progress_label=""
    ):
        """Calculates the per-log score used for anomaly decisions.

        In causal LM mode this returns token-prediction accuracies for each top-k.
        In sequence-classification mode this returns the anomaly probability and
        mirrors it into each configured `top{k}` column for compatibility with
        the rest of the reporting pipeline.
        """
        print(
            f"Calculating top-k accuracy{(' for ' + progress_label) if progress_label else ''}..."
        )
        accuracies = {f"top{k}": [] for k in self.config["top_k_values"]}
        model.to(self.device)
        model.eval()

        method = self.config.get("accuracy_method", "shifted")
        batch_size = max(1, int(self.config.get("eval_batch_size", 1)))
        max_len = int(
            self.config.get("eval_max_length", self.config.get("max_length", 1024))
        )
        use_autocast = bool(self.config.get("eval_use_autocast", True))

        total_texts = len(texts)
        with torch.no_grad():
            for start in range(0, total_texts, batch_size):
                end = min(start + batch_size, total_texts)
                if end % max(100, batch_size * 10) == 0 or end == total_texts:
                    label = f" ({progress_label})" if progress_label else ""
                    print(
                        f"\r  Calculating accuracy{label}... {end}/{total_texts}",
                        end="",
                    )

                inputs = tokenizer(
                    [str(text) for text in texts[start:end]],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get(
                    "attention_mask",
                    torch.ones_like(input_ids, device=input_ids.device),
                )

                if torch.cuda.is_available() and use_autocast:
                    autocast_dtype = (
                        torch.bfloat16
                        if str(self.config.get("eval_torch_dtype", "")).lower().strip()
                        in ("bfloat16", "bf16")
                        else torch.float16
                    )
                    autocast_ctx = torch.autocast(
                        device_type="cuda", dtype=autocast_dtype
                    )
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    outputs = model(**inputs)
                logits = outputs.logits

                if self._is_sequence_classification():
                    if logits.shape[-1] == 1:
                        anomaly_scores = torch.sigmoid(logits.squeeze(-1))
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        anomaly_scores = probs[:, self._anomaly_label_id()]
                    anomaly_scores = anomaly_scores.detach().float().cpu().tolist()
                    for k in self.config["top_k_values"]:
                        accuracies[f"top{k}"].extend(anomaly_scores)
                    continue

                if method in ("original", "legacy_flat"):
                    # "original": project/paper behavior (position-by-position).
                    # "legacy_flat": matches legado/federated_evaluation.py (membership check against the
                    # whole top_k[0] matrix, i.e., flattened over seq_len x k).
                    for k in self.config["top_k_values"]:
                        top_k_preds_indices = torch.topk(logits, k, dim=-1).indices

                        if method == "legacy_flat":
                            top_k_preds_indices = top_k_preds_indices.cpu().numpy()
                            input_tokens = input_ids.cpu().numpy()
                            valid_mask = attention_mask.cpu().numpy().astype(bool)
                            for sample_idx in range(len(input_tokens)):
                                valid_tokens = input_tokens[sample_idx][
                                    valid_mask[sample_idx]
                                ]
                                correct_predictions = sum(
                                    self._is_in_top_k(
                                        top_k_preds_indices[sample_idx], token
                                    )
                                    for token in valid_tokens
                                )
                                total_tokens = len(valid_tokens)
                                accuracies[f"top{k}"].append(
                                    correct_predictions / total_tokens
                                    if total_tokens > 0
                                    else 0
                                )
                        else:
                            matches = (
                                top_k_preds_indices == input_ids.unsqueeze(-1)
                            ).any(dim=-1)
                            valid_mask = attention_mask.bool()
                            correct_predictions = (
                                matches & valid_mask
                            ).sum(dim=-1).float()
                            total_tokens = valid_mask.sum(dim=-1).clamp_min(1).float()
                            accuracies[f"top{k}"].extend(
                                (correct_predictions / total_tokens).cpu().tolist()
                            )
                else:
                    # Método "shifted": próximo-token (mais canônico para LM)
                    shifted_logits = logits[..., :-1, :].contiguous()
                    labels = input_ids[..., 1:].contiguous()
                    valid_mask = attention_mask[..., 1:].bool()

                    for k in self.config["top_k_values"]:
                        top_k_preds = torch.topk(shifted_logits, k, dim=-1).indices

                        # Check if the true next token is in the top-k predictions
                        matches = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1)
                        correct = (matches & valid_mask).sum(dim=-1).float()
                        total = valid_mask.sum(dim=-1).clamp_min(1).float()

                        accuracies[f"top{k}"].extend(
                            (correct / total).cpu().tolist()
                        )

        print()

        return pd.DataFrame(accuracies)

    def _calculate_top_k_accuracy(self, model, tokenizer):
        """Convenience wrapper to evaluate on the configured test set."""
        return self._calculate_top_k_accuracy_for_texts(
            model,
            tokenizer,
            self.test_df["Content"].astype(str).tolist(),
            progress_label="test",
        )

    def _select_threshold_f1_max(self, scores, labels):
        thresholds = np.linspace(0, 1, int(self.config["f1_threshold_steps"]))
        best_f1 = -1.0
        best_th = 0.0
        for th in thresholds:
            preds = self._predict_from_scores(scores, th)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
        return best_th, float(best_f1)

    def _select_threshold_fpr_target(self, benign_scores):
        target = float(self.config.get("fpr_target", 0.01))
        target = min(max(target, 0.0), 1.0)
        scores = np.asarray(benign_scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return 0.0

        if self._score_descending_is_anomaly():
            # Want P(score >= th) ~= target for benign-only calibration set.
            threshold = float(np.quantile(scores, 1.0 - target))
            if threshold <= 0.0:
                positive_scores = scores[scores > 0.0]
                if positive_scores.size:
                    # Keep the operational rule numerically valid even when
                    # reduced-precision inference collapses the quantile to zero.
                    return float(np.min(positive_scores))
            return threshold
        # Want P(score < th) ~= target for benign-only calibration set.
        return float(np.quantile(scores, target))

    def _benchmark_scores_from_outputs(
        self, outputs, input_ids, attention_mask, benchmark_k: int
    ):
        logits = outputs.logits

        if self._is_sequence_classification():
            if logits.shape[-1] == 1:
                scores = torch.sigmoid(logits.squeeze(-1))
            else:
                probs = torch.softmax(logits, dim=-1)
                scores = probs[:, self._anomaly_label_id()]
            return scores.detach().float().cpu().numpy()

        method = self.config.get("accuracy_method", "shifted")

        if method in ("original", "legacy_flat"):
            top_k_preds_indices = torch.topk(logits, benchmark_k, dim=-1).indices
            if method == "legacy_flat":
                top_k_preds_indices = top_k_preds_indices.cpu().numpy()
                input_tokens = input_ids.cpu().numpy()
                valid_mask = attention_mask.cpu().numpy().astype(bool)
                scores = []
                for sample_idx in range(len(input_tokens)):
                    valid_tokens = input_tokens[sample_idx][valid_mask[sample_idx]]
                    correct_predictions = sum(
                        self._is_in_top_k(top_k_preds_indices[sample_idx], token)
                        for token in valid_tokens
                    )
                    total_tokens = len(valid_tokens)
                    scores.append(
                        correct_predictions / total_tokens if total_tokens > 0 else 0.0
                    )
                return np.asarray(scores, dtype=np.float32)

            matches = (top_k_preds_indices == input_ids.unsqueeze(-1)).any(dim=-1)
            valid_mask = attention_mask.bool()
            correct_predictions = (matches & valid_mask).sum(dim=-1).float()
            total_tokens = valid_mask.sum(dim=-1).clamp_min(1).float()
            return (correct_predictions / total_tokens).detach().cpu().numpy()

        shifted_logits = logits[..., :-1, :].contiguous()
        labels = input_ids[..., 1:].contiguous()
        valid_mask = attention_mask[..., 1:].bool()
        top_k_preds = torch.topk(shifted_logits, benchmark_k, dim=-1).indices
        matches = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1)
        correct = (matches & valid_mask).sum(dim=-1).float()
        total = valid_mask.sum(dim=-1).clamp_min(1).float()
        return (correct / total).detach().cpu().numpy()

    def _benchmark_inference(self, model, tokenizer, texts, *, round_num):
        if not texts:
            return None

        warmup = int(self.config.get("benchmark_warmup", 10))
        max_len = int(
            self.config.get("eval_max_length", self.config.get("max_length", 1024))
        )
        configured_batch_sizes = self.config.get("benchmark_batch_sizes")
        if configured_batch_sizes:
            batch_sizes = sorted(
                {
                    max(1, int(batch_size))
                    for batch_size in configured_batch_sizes
                    if int(batch_size) > 0
                }
            )
        else:
            batch_sizes = sorted(
                {
                    1,
                    max(1, int(self.config.get("eval_batch_size", 1))),
                }
            )

        benchmark_threshold = float(self.config.get("benchmark_threshold", 0.5))
        benchmark_k = int(self.config.get("benchmark_k", self.config["top_k_values"][0]))
        use_autocast = bool(self.config.get("eval_use_autocast", True))
        timed_texts = [str(text) for text in texts[warmup:]]
        warmup_texts = [str(text) for text in texts[:warmup]]
        all_results = []

        model.to(self.device)
        model.eval()

        for batch_size in batch_sizes:
            e2e_per_sample_ms = []
            forward_per_sample_ms = []
            batch_elapsed_ms = []

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                for start_idx in range(0, len(warmup_texts), batch_size):
                    batch_texts = warmup_texts[start_idx : start_idx + batch_size]
                    if not batch_texts:
                        continue
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    if torch.cuda.is_available() and use_autocast:
                        autocast_dtype = (
                            torch.bfloat16
                            if str(self.config.get("eval_torch_dtype", "")).lower().strip()
                            in ("bfloat16", "bf16")
                            else torch.float16
                        )
                        autocast_ctx = torch.autocast(
                            device_type="cuda", dtype=autocast_dtype
                        )
                    else:
                        autocast_ctx = nullcontext()

                    with autocast_ctx:
                        _ = model(**inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                for start_idx in range(0, len(timed_texts), batch_size):
                    batch_texts = timed_texts[start_idx : start_idx + batch_size]
                    if not batch_texts:
                        continue

                    start_total = time.perf_counter()
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_forward = time.perf_counter()

                    if torch.cuda.is_available() and use_autocast:
                        autocast_dtype = (
                            torch.bfloat16
                            if str(self.config.get("eval_torch_dtype", "")).lower().strip()
                            in ("bfloat16", "bf16")
                            else torch.float16
                        )
                        autocast_ctx = torch.autocast(
                            device_type="cuda", dtype=autocast_dtype
                        )
                    else:
                        autocast_ctx = nullcontext()

                    with autocast_ctx:
                        outputs = model(**inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_forward = time.perf_counter()

                    scores = self._benchmark_scores_from_outputs(
                        outputs,
                        inputs["input_ids"],
                        inputs.get(
                            "attention_mask",
                            torch.ones_like(inputs["input_ids"], device=inputs["input_ids"].device),
                        ),
                        benchmark_k,
                    )
                    _ = self._predict_from_scores(scores, benchmark_threshold)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_total = time.perf_counter()

                    batch_count = len(batch_texts)
                    e2e_ms = (end_total - start_total) * 1000.0
                    forward_ms = (end_forward - start_forward) * 1000.0
                    batch_elapsed_ms.append(e2e_ms)
                    e2e_per_sample_ms.extend([e2e_ms / batch_count] * batch_count)
                    forward_per_sample_ms.extend([forward_ms / batch_count] * batch_count)

            if torch.cuda.is_available():
                peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            else:
                peak_mb = float("nan")

            timed_samples = len(e2e_per_sample_ms)
            total_e2e_seconds = float(sum(batch_elapsed_ms) / 1000.0)
            throughput = (
                float(timed_samples / total_e2e_seconds)
                if total_e2e_seconds > 0
                else float("nan")
            )

            def _summary(values, percentile):
                if not values:
                    return float("nan")
                return float(np.percentile(values, percentile))

            all_results.append(
                {
                    "round": int(round_num),
                    "model_name": self.config.get("model_name", ""),
                    "eval_max_length": max_len,
                    "num_samples": int(len(texts)),
                    "warmup": warmup,
                    "timed_samples": int(timed_samples),
                    "benchmark_batch_size": int(batch_size),
                    "benchmark_k": int(benchmark_k),
                    "benchmark_threshold": float(benchmark_threshold),
                    "mean_e2e_ms_per_sample": _summary(e2e_per_sample_ms, 50)
                    if len(e2e_per_sample_ms) == 1
                    else float(np.mean(e2e_per_sample_ms))
                    if e2e_per_sample_ms
                    else float("nan"),
                    "median_e2e_ms_per_sample": _summary(e2e_per_sample_ms, 50),
                    "p95_e2e_ms_per_sample": _summary(e2e_per_sample_ms, 95),
                    "p99_e2e_ms_per_sample": _summary(e2e_per_sample_ms, 99),
                    "mean_forward_ms_per_sample": float(np.mean(forward_per_sample_ms))
                    if forward_per_sample_ms
                    else float("nan"),
                    "median_forward_ms_per_sample": _summary(forward_per_sample_ms, 50),
                    "p95_forward_ms_per_sample": _summary(forward_per_sample_ms, 95),
                    "p99_forward_ms_per_sample": _summary(forward_per_sample_ms, 99),
                    "throughput_samples_per_second": throughput,
                    "peak_cuda_memory_mb": peak_mb,
                    "fedprox_mu": float(self.config.get("fedprox_mu", 0.0)),
                    "aggregation_method": "FedProx"
                    if self.config.get("fedprox_mu", 0.0) > 0
                    else "FedAvg",
                }
            )

        return all_results

    def _compute_temporal_metrics_from_df(self, df, round_num, k, *, granularity):
        """
        Computes early-stage detection metrics (TTD, coverage, FPR) when
        temporal and device information is available in the test set.

        - TTD (Time-to-Detection): tempo médio/mediano entre o início do ataque
          em um dispositivo e a primeira detecção.
        - Detection coverage: fração de dispositivos atacados para os quais
          houve detecção.
        - Benign FPR: taxa de falsos positivos em tráfego benigno.
        """
        required_cols = ["Timestamp_parsed", "Src IP", "Label", "pred"]
        if not all(col in df.columns for col in required_cols):
            return None

        df = df.dropna(subset=["Timestamp_parsed"]).copy()

        ttds = []
        detection_coverage = 0
        attacked_devices = 0

        # Métricas por dispositivo (Src IP)
        for device_id, group in df.groupby("Src IP"):
            group = group.sort_values("Timestamp_parsed")
            if (group["Label"] == 1).any():
                attacked_devices += 1
                attack_start_time = group.loc[
                    group["Label"] == 1, "Timestamp_parsed"
                ].min()
                detections = group[
                    (group["Timestamp_parsed"] >= attack_start_time)
                    & (group["pred"] == 1)
                ]
                if not detections.empty:
                    first_detection_time = detections["Timestamp_parsed"].min()
                    delta = (first_detection_time - attack_start_time).total_seconds()
                    ttds.append(max(delta, 0.0))
                    detection_coverage += 1

        if attacked_devices > 0 and ttds:
            mean_ttd = float(np.mean(ttds))
            median_ttd = float(np.median(ttds))
            detection_coverage_ratio = detection_coverage / attacked_devices
        else:
            mean_ttd = float("nan")
            median_ttd = float("nan")
            detection_coverage_ratio = 0.0

        # FPR em tráfego benigno
        benign = df[df["Label"] == 0]
        if not benign.empty:
            benign_fpr = float((benign["pred"] == 1).mean())
        else:
            benign_fpr = float("nan")

        return {
            "round": round_num,
            "k": k,
            "granularity": granularity,
            "mean_ttd_seconds": mean_ttd,
            "median_ttd_seconds": median_ttd,
            "detection_coverage": detection_coverage_ratio,
            "benign_fpr": benign_fpr,
            "num_attacked_devices": attacked_devices,
            # FedProx parameters for comparison
            "fedprox_mu": float(self.config.get("fedprox_mu", 0.0)),
            "aggregation_method": "FedProx"
            if self.config.get("fedprox_mu", 0.0) > 0
            else "FedAvg",
        }

    def _compute_temporal_metrics(self, preds, round_num, k):
        required_cols = ["Timestamp", "Src IP", "Label"]
        if not all(col in self.test_df.columns for col in required_cols):
            print("Temporal metrics skipped: required columns not found in test.csv.")
            return None

        df = self.test_df[["Timestamp", "Src IP", "Label"]].copy()
        df["pred"] = preds.values if hasattr(preds, "values") else preds
        df["Timestamp_parsed"] = self._parse_timestamps(df["Timestamp"])
        return self._compute_temporal_metrics_from_df(
            df, round_num, k, granularity="flow"
        )

    def _window_aggregate(self, series):
        agg = str(self.config.get("temporal_window_agg", "mean")).lower()
        if series.empty:
            return float("nan")
        if agg == "max":
            return float(series.max())
        if agg == "min":
            return float(series.min())
        if agg == "p90":
            return float(np.quantile(series.values, 0.90))
        if agg == "p10":
            return float(np.quantile(series.values, 0.10))
        if agg == "p25":
            return float(np.quantile(series.values, 0.25))
        return float(series.mean())

    def _build_windowed_df(self, base_df, score_series):
        """
        Builds a window-aggregated dataframe for temporal evaluation.

        base_df must contain: Timestamp, Src IP, Label (same row order as score_series).
        Returns a dataframe with: Timestamp_parsed, Src IP, Label, score.
        """
        window_seconds = int(self.config.get("temporal_window_seconds", 30))
        if window_seconds <= 0:
            return None

        df = base_df[["Timestamp", "Src IP", "Label"]].copy()
        df["score"] = score_series.values
        df["Timestamp_parsed"] = self._parse_timestamps(df["Timestamp"])
        df = df.dropna(subset=["Timestamp_parsed"])

        # Group by device and time bin
        grouper = pd.Grouper(key="Timestamp_parsed", freq=f"{window_seconds}S")
        grouped = df.groupby(["Src IP", grouper], dropna=False)

        out = grouped.agg(
            Label=("Label", "max"),
            score=("score", lambda s: self._window_aggregate(s)),
        ).reset_index()
        # Timestamp_parsed here is the window start as produced by the Grouper
        return out

    def evaluate(self):
        """
        Main evaluation loop. Iterates through saved models, calculates metrics,
        and saves results.
        """
        print("--- Starting Evaluation ---")
        tokenizer = self._build_tokenizer()
        self._cached_tokenizer_pad_id = tokenizer.pad_token_id

        all_f1_results = []
        all_temporal_results = []
        all_bench_results = []
        print(f"Evaluating {self.config['num_rounds']} rounds...")

        rounds = list(range(1, int(self.config["num_rounds"]) + 1))
        if self.config.get("include_round_0", False):
            rounds = [0] + rounds

        benchmark_rounds = set()
        if self.config.get("benchmark_inference", False):
            for r in self.config.get("benchmark_rounds", []):
                try:
                    benchmark_rounds.add(int(r))
                except Exception:
                    continue
            if benchmark_rounds:
                rounds = sorted(set(rounds).union(benchmark_rounds))

        for round_num in rounds:
            print(f"\n--- Evaluating Round {round_num} ---")
            model_path = os.path.join(
                self.results_dir, f"round_{round_num}", "global_model"
            )

            if not os.path.exists(model_path):
                print(f"Model for round {round_num} not found. Skipping.")
                continue

            print("Loading model...")
            model = self._load_model_from_path(model_path)

            # Optional inference benchmark (deployability)
            if (
                self.config.get("benchmark_inference", False)
                and int(round_num) in benchmark_rounds
            ):
                n = int(self.config.get("benchmark_num_samples", 200))
                texts = self.test_df["Content"].astype(str).tolist()
                if n > 0 and len(texts) > n:
                    texts = texts[:n]
                bench = self._benchmark_inference(
                    model, tokenizer, texts, round_num=round_num
                )
                if bench is not None:
                    if isinstance(bench, list):
                        all_bench_results.extend(bench)
                    else:
                        all_bench_results.append(bench)

            print("Calculating accuracies...")
            # Calculate accuracies
            acc_df = self._calculate_top_k_accuracy(model, tokenizer)
            print("Accuracies calculated:")
            acc_df["label"] = self.test_df["Label"]
            print(acc_df.describe())

            threshold_mode = self._get_threshold_mode()
            calib_acc_df = None
            calib_window_df_by_k = {}
            if threshold_mode == "fpr_target":
                calib_df = self._load_calibration_data()
                calib_acc_df = self._calculate_top_k_accuracy_for_texts(
                    model,
                    tokenizer,
                    calib_df["Content"].astype(str).tolist(),
                    progress_label="calibration",
                )
                if (
                    str(self.config.get("temporal_eval_mode", "flow")).lower()
                    == "window"
                ):
                    for k in self.config["top_k_values"]:
                        calib_window_df_by_k[k] = self._build_windowed_df(
                            calib_df, calib_acc_df[f"top{k}"]
                        )

            # Compute metrics for each k
            for k in self.config["top_k_values"]:
                scores = acc_df[f"top{k}"]

                temporal_mode = str(
                    self.config.get("temporal_eval_mode", "flow")
                ).lower()
                if temporal_mode == "window":
                    # Windowed evaluation (aggregate per Src IP and time bin)
                    window_df = self._build_windowed_df(self.test_df, scores)
                    if window_df is None or window_df.empty:
                        print(
                            "  Windowed evaluation skipped: unable to build windows (missing Timestamp/Src IP?)."
                        )
                        continue

                    if threshold_mode == "fpr_target":
                        calib_window_df = calib_window_df_by_k.get(k)
                        th = self._select_threshold_fpr_target(
                            calib_window_df["score"].dropna().values
                            if calib_window_df is not None
                            else np.array([])
                        )
                    else:
                        th, _ = self._select_threshold_f1_max(
                            window_df["score"].values, window_df["Label"].values
                        )

                    window_df["pred"] = self._predict_from_scores(
                        window_df["score"].values, th
                    )
                    f1 = float(f1_score(window_df["Label"], window_df["pred"]))
                    precision = float(
                        precision_score(
                            window_df["Label"], window_df["pred"], zero_division=0
                        )
                    )
                    recall = float(
                        recall_score(
                            window_df["Label"], window_df["pred"], zero_division=0
                        )
                    )
                    benign_fpr = (
                        float(
                            (window_df.loc[window_df["Label"] == 0, "pred"] == 1).mean()
                        )
                        if (window_df["Label"] == 0).any()
                        else float("nan")
                    )

                    print(
                        f"  K={k} (window): F1={f1:.4f} | Threshold={th:.4f} ({threshold_mode})"
                        f" | Precision={precision:.4f}, Recall={recall:.4f}, BenignFPR={benign_fpr:.4f}"
                    )

                    all_f1_results.append(
                        {
                            "round": round_num,
                            "k": k,
                            "granularity": "window",
                            "window_seconds": int(
                                self.config.get("temporal_window_seconds", 30)
                            ),
                            "window_agg": str(
                                self.config.get("temporal_window_agg", "mean")
                            ),
                            "threshold_mode": threshold_mode,
                            "fpr_target": float(self.config.get("fpr_target", np.nan))
                            if threshold_mode == "fpr_target"
                            else np.nan,
                            "calibration_source": self.config.get(
                                "calibration_source", ""
                            )
                            if threshold_mode == "fpr_target"
                            else "",
                            "calibration_num_samples": int(
                                self.config.get("calibration_num_samples", 0)
                            )
                            if threshold_mode == "fpr_target"
                            else 0,
                            "threshold": th,
                            "f1_score": f1,
                            "precision": precision,
                            "recall": recall,
                            "benign_fpr": benign_fpr,
                            "num_windows": int(len(window_df)),
                            # FedProx parameters for comparison
                            "fedprox_mu": float(self.config.get("fedprox_mu", 0.0)),
                            "aggregation_method": "FedProx"
                            if self.config.get("fedprox_mu", 0.0) > 0
                            else "FedAvg",
                        }
                    )

                    if self.config.get("enable_temporal_metrics", True):
                        # Reuse temporal metric logic with window start timestamps
                        tmp = window_df.rename(
                            columns={"Timestamp_parsed": "Timestamp_parsed"}
                        ).copy()
                        temporal_metrics = self._compute_temporal_metrics_from_df(
                            tmp, round_num, k, granularity="window"
                        )
                        if temporal_metrics is not None:
                            temporal_metrics["window_seconds"] = int(
                                self.config.get("temporal_window_seconds", 30)
                            )
                            temporal_metrics["window_agg"] = str(
                                self.config.get("temporal_window_agg", "mean")
                            )
                            all_temporal_results.append(temporal_metrics)
                else:
                    # Flow-level evaluation (current behavior)
                    if threshold_mode == "fpr_target":
                        th = self._select_threshold_fpr_target(
                            calib_acc_df[f"top{k}"].values
                        )
                        preds = self._predict_from_scores(scores.values, th)
                        f1 = float(f1_score(acc_df["label"], preds))
                    else:
                        th, f1 = self._select_threshold_f1_max(
                            scores.values, acc_df["label"].values
                        )
                        preds = self._predict_from_scores(scores.values, th)

                    precision = float(
                        precision_score(acc_df["label"], preds, zero_division=0)
                    )
                    recall = float(
                        recall_score(acc_df["label"], preds, zero_division=0)
                    )
                    benign_fpr = (
                        float((preds[acc_df["label"] == 0] == 1).mean())
                        if (acc_df["label"] == 0).any()
                        else float("nan")
                    )

                    print(
                        f"  K={k}: F1={f1:.4f} | Threshold={th:.4f} ({threshold_mode})"
                        f" | Precision={precision:.4f}, Recall={recall:.4f}, BenignFPR={benign_fpr:.4f}"
                    )

                    all_f1_results.append(
                        {
                            "round": round_num,
                            "k": k,
                            "granularity": "flow",
                            "threshold_mode": threshold_mode,
                            "fpr_target": float(self.config.get("fpr_target", np.nan))
                            if threshold_mode == "fpr_target"
                            else np.nan,
                            "calibration_source": self.config.get(
                                "calibration_source", ""
                            )
                            if threshold_mode == "fpr_target"
                            else "",
                            "calibration_num_samples": int(
                                self.config.get("calibration_num_samples", 0)
                            )
                            if threshold_mode == "fpr_target"
                            else 0,
                            "threshold": th,
                            "f1_score": f1,
                            "precision": precision,
                            "recall": recall,
                            "benign_fpr": benign_fpr,
                            # FedProx parameters for comparison
                            "fedprox_mu": float(self.config.get("fedprox_mu", 0.0)),
                            "aggregation_method": "FedProx"
                            if self.config.get("fedprox_mu", 0.0) > 0
                            else "FedAvg",
                        }
                    )

                    if self.config.get("enable_temporal_metrics", True):
                        temporal_metrics = self._compute_temporal_metrics(
                            preds, round_num, k
                        )
                        if temporal_metrics is not None:
                            all_temporal_results.append(temporal_metrics)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save final results
        f1_df = pd.DataFrame(all_f1_results)
        f1_results_path = self._output_csv_path("f1_scores")
        f1_df.to_csv(f1_results_path, index=False)
        print(f"\nEvaluation complete. F1 score results saved to {f1_results_path}")

        # Salva métricas temporais, se tiverem sido calculadas
        if all_temporal_results:
            temporal_df = pd.DataFrame(all_temporal_results)
            temporal_results_path = self._output_csv_path("temporal_metrics")
            temporal_df.to_csv(temporal_results_path, index=False)
            print(
                f"Temporal evaluation complete. Results saved to {temporal_results_path}"
            )

        if all_bench_results:
            bench_df = pd.DataFrame(all_bench_results)
            bench_path = self._output_csv_path("inference_benchmark")
            bench_df.to_csv(bench_path, index=False)
            print(f"Inference benchmark complete. Results saved to {bench_path}")
