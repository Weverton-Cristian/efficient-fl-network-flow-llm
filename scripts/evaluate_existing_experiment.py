#!/usr/bin/env python3
import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Runs only the evaluator for an existing Efficient FL experiment, "
            "optionally with operational FPR calibration and latency benchmarking."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config_wifi.yaml",
        help="Base configuration file.",
    )
    parser.add_argument(
        "--simulation_name",
        default=None,
        help="Override simulation_name if you want to evaluate another results directory.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Optional override for num_rounds. Use 0 with --benchmark_rounds to evaluate only benchmark rounds.",
    )
    parser.add_argument(
        "--evaluation_output_dir",
        default=None,
        help="Optional directory where evaluation CSVs should be written.",
    )
    parser.add_argument(
        "--threshold_selection",
        default="fpr_target",
        choices=["fpr_target", "f1_max"],
        help="Threshold policy for evaluation.",
    )
    parser.add_argument("--fpr_target", type=float, default=0.10, help="Target benign FPR.")
    parser.add_argument(
        "--benchmark_inference",
        action="store_true",
        help="Also export inference_benchmark*.csv for the selected rounds.",
    )
    parser.add_argument(
        "--benchmark_rounds",
        default="",
        help="Comma-separated rounds to benchmark. Example: 1,6,12",
    )
    parser.add_argument(
        "--benchmark_num_samples",
        type=int,
        default=200,
        help="Number of samples used in inference benchmarking.",
    )
    parser.add_argument(
        "--benchmark_warmup",
        type=int,
        default=10,
        help="Warmup samples before timing inference.",
    )
    parser.add_argument(
        "--benchmark_batch_sizes",
        default="",
        help="Comma-separated benchmark batch sizes. Example: 1,8",
    )
    parser.add_argument(
        "--calibration_num_samples",
        type=int,
        default=None,
        help="Optional override for benign calibration samples used by fpr_target.",
    )
    parser.add_argument(
        "--eval_samples_per_class",
        type=int,
        default=0,
        help="Optional balanced cap per class for faster comparable validation runs. Use 0 for full evaluation.",
    )
    parser.add_argument(
        "--top_k_values",
        default="",
        help="Optional comma-separated top-k values override. Example: 1 or 1,5",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Optional eval batch size override.")
    parser.add_argument(
        "--eval_torch_dtype",
        default=None,
        choices=["float32", "float16", "bfloat16", None],
        help="Optional inference dtype override.",
    )
    parser.add_argument(
        "--eval_use_autocast",
        action="store_true",
        help="Enable autocast during evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src.evaluation.evaluator import Evaluator
    from src.utils.hf import apply_hf_environment

    config_path = (project_root / args.config).resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)

    if args.simulation_name:
        config["simulation_name"] = args.simulation_name
    if args.num_rounds is not None:
        config["num_rounds"] = int(args.num_rounds)
    if args.evaluation_output_dir:
        config["evaluation_output_dir"] = args.evaluation_output_dir
    config["threshold_selection"] = args.threshold_selection
    config["fpr_target"] = float(args.fpr_target)
    config["benchmark_inference"] = bool(args.benchmark_inference)
    config["benchmark_rounds"] = parse_int_list(args.benchmark_rounds)
    config["benchmark_num_samples"] = int(args.benchmark_num_samples)
    config["benchmark_warmup"] = int(args.benchmark_warmup)
    config["benchmark_batch_sizes"] = parse_int_list(args.benchmark_batch_sizes)
    if args.top_k_values:
        config["top_k_values"] = parse_int_list(args.top_k_values)
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = int(args.eval_batch_size)
    if args.eval_torch_dtype is not None:
        config["eval_torch_dtype"] = args.eval_torch_dtype
    if args.eval_use_autocast:
        config["eval_use_autocast"] = True
    if args.calibration_num_samples is not None:
        config["calibration_num_samples"] = int(args.calibration_num_samples)

    os.chdir(project_root)
    apply_hf_environment(config)
    set_seed(int(args.seed))

    print(f"Loaded config: {config_path}")
    print(f"Evaluating simulation: {config['simulation_name']}")
    if config.get("evaluation_output_dir"):
        print(f"Evaluation output dir: {config['evaluation_output_dir']}")

    evaluator = Evaluator(config)
    if int(args.eval_samples_per_class) > 0:
        anomaly_df = evaluator.test_df[evaluator.test_df["Label"] == 1].copy()
        benign_df = evaluator.test_df[evaluator.test_df["Label"] == 0].copy()
        sample_size = min(
            len(anomaly_df),
            len(benign_df),
            int(args.eval_samples_per_class),
        )
        if sample_size > 0:
            anomaly_df = anomaly_df.sample(n=sample_size, random_state=int(args.seed))
            benign_df = benign_df.sample(n=sample_size, random_state=int(args.seed))
            evaluator.test_df = (
                pd.concat([anomaly_df, benign_df], ignore_index=True)
                .sample(frac=1.0, random_state=int(args.seed))
                .reset_index(drop=True)
            )
        print(f"Test distribution used: {evaluator.test_df['Label'].value_counts().to_dict()}")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
