import yaml
import argparse
import os
import copy
import random
import numpy as np
import torch
from src.data_processing.wifi_processor import WiFiProcessor
from src.federated_learning.server import FederatedServer
from src.evaluation.evaluator import Evaluator
from src.utils.hf import apply_hf_environment

def _merge_overrides(base: dict, overrides: dict) -> dict:
    """
    Shallow-merge overrides into a copy of base config.
    Intended for evaluation-only overrides (thresholding, temporal settings, etc.).
    """
    merged = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        merged[k] = v
    return merged


def _set_global_seed(config: dict) -> None:
    """
    Sets the main-process random seed for better run-to-run reproducibility.

    This does not make every CUDA kernel fully deterministic, but it does fix the
    project-level stochastic sources we control directly: Python, NumPy and torch.
    """
    seed = int(config.get("random_seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}.")

def main(config_path, config_overrides=None):
    """
    Main function to orchestrate the federated learning pipeline.
    """
    # 1. Load Configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    config = _merge_overrides(config, config_overrides)

    print("Configuration loaded successfully.")
    print(f"Starting simulation: {config.get('simulation_name', 'N/A')}")

    # Configure Hugging Face cache/offline behavior before any tokenizer/model loads.
    apply_hf_environment(config)
    _set_global_seed(config)

    # 2. Execute Data Processing Pipeline
    # This repository intentionally supports only the WiFi/network-flow pipeline.
    print("\n--- Initializing Data Processing ---")
    if config.get("dataset_name") != "wifi":
        raise ValueError(
            "This repository supports dataset_name='wifi' only. "
            "Legacy datasets remain in the article repository."
        )
    processor = WiFiProcessor(config)

    processor.run()

    # 3. Execute Federated Training
    print("\n--- Starting Federated Training ---")
    server = FederatedServer(config)
    server.run_federated_training()
    print("--- Federated Training Complete ---")

    # 4. Execute Evaluation
    print("\n--- Starting Evaluation ---")
    evaluator = Evaluator(config)
    evaluator.evaluate()
    print("--- Evaluation Complete ---")

    # 5. Optional additional evaluation runs (e.g., operational fpr_target + temporal metrics)
    evaluation_runs = config.get("evaluation_runs", [])
    if evaluation_runs:
        print("\n--- Starting Additional Evaluations (Overrides) ---")
        for i, run_cfg in enumerate(evaluation_runs, start=1):
            name = run_cfg.get("name", f"eval_{i}")
            overrides = run_cfg.get("overrides", {})
            eval_config = _merge_overrides(config, overrides)
            print(f"\n--- Evaluation Override {i}/{len(evaluation_runs)}: {name} ---")
            evaluator = Evaluator(eval_config)
            evaluator.evaluate()
        print("--- Additional Evaluations Complete ---")

import multiprocessing as mp

def _parse_cli_overrides(raw_overrides: list[str] | None) -> dict:
    """
    Parses flat KEY=VALUE overrides using YAML semantics for values.

    Example:
      --set hf_offline=false --set calibration_num_samples=1000
    """
    parsed = {}
    for raw in raw_overrides or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{raw}'. Empty key.")
        parsed[key] = yaml.safe_load(value)
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run efficient federated learning for WiFi network-flow analysis."
    )
    parser.add_argument('--config', type=str, default='configs/config_wifi.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a top-level YAML value. Can be used multiple times.",
    )
    args = parser.parse_args()

    # Define o método de início de multiprocessamento como 'spawn'.
    # Isso é crucial para evitar erros de inicialização da CUDA em processos filhos.
    try:
        mp.set_start_method('spawn', force=True)
        print("Método de início de multiprocessamento configurado para 'spawn'.")
    except RuntimeError:
        # Pode já ter sido definido, o que não é um problema.
        pass

    # Ensure the script's working directory is the project root
    # This makes path handling in config.yaml consistent
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    cli_overrides = _parse_cli_overrides(args.overrides)
    main(args.config, cli_overrides)
