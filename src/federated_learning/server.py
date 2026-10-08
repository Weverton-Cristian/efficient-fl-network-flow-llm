import os
import random
import math
import json
import csv
import time
import torch
import numpy as np
from datasets import load_from_disk
import concurrent.futures

from src.models.model_loader import initialize_global_model
from .client import ClientTrainer


def format_time(seconds):
    """Formata segundos em formato legível (HH:MM:SS)."""
    if seconds < 0:
        return "--:--:--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {secs:02d}s"
    else:
        return f"{secs:02d}s"


def train_client_process(args):
    """
    Função wrapper para treinar um cliente em um processo separado.
    Isso é necessário para o paralelismo com ProcessPoolExecutor.
    """
    client_id, config, round_num, learning_rate, gpu_id = args

    # Derive a stable per-client/per-round seed so local training remains
    # reproducible even when clients run in spawned worker processes.
    base_seed = int(config.get("random_seed", 42))
    derived_seed = base_seed + (int(round_num) * 1000) + int(client_id)
    random.seed(derived_seed)
    np.random.seed(derived_seed)
    torch.manual_seed(derived_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived_seed)

    # Define qual GPU este processo deve usar
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    print(f"Iniciando treinamento para o cliente {client_id} na GPU {gpu_id}")
    # Passa o gpu_id para o ClientTrainer
    client_trainer = ClientTrainer(client_id, config, gpu_id)
    cpu_weights = client_trainer.train(round_num, learning_rate)
    return cpu_weights


class FederatedServer:
    """
    Orchestrates the federated learning process, including client selection,
    training, and model aggregation.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model, self.tokenizer = initialize_global_model(config)
        self.global_model.to(self.device)

        # Mantém metadados sobre o tamanho de cada shard de cliente para
        # permitir seleção adaptativa e agregação ponderada.
        self.client_sample_counts = {}

        # Métricas de custo de comunicação por rodada (tamanho dos updates)
        self.communication_metrics = []

    def _weights_size_bytes(self, weights_dict):
        """Returns total serialized size in bytes for a state_dict-like object."""
        total = 0
        if not weights_dict:
            return 0
        for v in weights_dict.values():
            if isinstance(v, torch.Tensor):
                total += int(v.numel()) * int(v.element_size())
        return int(total)

    def _weights_num_params(self, weights_dict):
        total = 0
        if not weights_dict:
            return 0
        for v in weights_dict.values():
            if isinstance(v, torch.Tensor):
                total += int(v.numel())
        return int(total)

    def _record_round_communication(self, round_num, client_ids, client_updates):
        sizes = [self._weights_size_bytes(u) for u in client_updates]
        params = [self._weights_num_params(u) for u in client_updates]
        total_bytes = int(sum(sizes))
        total_params = int(sum(params))

        if sizes:
            sizes_sorted = sorted(sizes)
            median_bytes = float(sizes_sorted[len(sizes_sorted) // 2])
            mean_bytes = float(total_bytes / len(sizes_sorted))
        else:
            median_bytes = float("nan")
            mean_bytes = float("nan")

        self.communication_metrics.append(
            {
                "round": int(round_num),
                "num_selected_clients": int(len(client_ids)),
                "lora": bool(self.config.get("lora", False)),
                "lora_rank": int(self.config.get("lora_rank", 0))
                if self.config.get("lora", False)
                else 0,
                "use_weighted_aggregation": bool(
                    self.config.get("use_weighted_aggregation", False)
                ),
                "client_selection_strategy": str(
                    self.config.get("client_selection_strategy", "uniform")
                ),
                "data_distribution_strategy": str(
                    self.config.get("data_distribution_strategy", "iid")
                ),
                # FedProx parameters
                "fedprox_mu": float(self.config.get("fedprox_mu", 0.0)),
                "aggregation_method": "FedProx"
                if self.config.get("fedprox_mu", 0.0) > 0
                else "FedAvg",
                "bytes_total": total_bytes,
                "bytes_mean_per_client": mean_bytes,
                "bytes_median_per_client": median_bytes,
                "params_total": total_params,
            }
        )

    def _save_communication_metrics(self):
        if not self.communication_metrics:
            return
        out_dir = os.path.join(
            self.config["results_path"], self.config["simulation_name"]
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "communication_metrics.csv")
        fieldnames = list(self.communication_metrics[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.communication_metrics)
        print(f"Communication metrics saved to: {out_path}")

    def _split_data_for_clients(self):
        """
        Splits the main tokenized dataset into shards for each client.
        Supports IID and Non-IID strategies and records per-client sample counts.
        This is a one-time setup operation per simulation.
        """
        legacy_layout = bool(self.config.get("legacy_flresults_layout", False))
        if legacy_layout:
            # Legacy scripts store per-client shards directly under round_0/.
            client_data_base_path = os.path.join(
                self.config["results_path"], self.config["simulation_name"], "round_0"
            )
            metadata_path = None
        else:
            client_data_base_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                "client_data",
            )
            metadata_path = os.path.join(
                client_data_base_path, "client_data_metadata.json"
            )

        # Avoid re-splitting if already done (only if shards exist).
        if os.path.exists(client_data_base_path):
            existing = []
            try:
                existing = os.listdir(client_data_base_path)
            except Exception:
                existing = []

            has_shards = any(name.startswith("client_") for name in existing)
            if has_shards:
                print("Client data shards already exist. Skipping split.")
                # Tenta carregar metadados previamente salvos para uso em seleção/agragação
                if metadata_path and os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            data = json.load(f)
                        # Converte chaves para int
                        self.client_sample_counts = {
                            int(k): int(v) for k, v in data.items()
                        }
                        print("Loaded client sample counts metadata.")
                    except Exception as e:
                        print(f"Warning: Failed to load client metadata: {e}")
                else:
                    # Best-effort: infere contagens carregando os datasets de cada cliente
                    print(
                        "Client metadata not found. Inferring sample counts from disk..."
                    )
                    self.client_sample_counts = {}
                    for client_id in range(self.config["num_clients"]):
                        client_path = os.path.join(
                            client_data_base_path, f"client_{client_id}"
                        )
                        if os.path.exists(client_path):
                            try:
                                client_dataset = load_from_disk(client_path)
                                self.client_sample_counts[client_id] = len(
                                    client_dataset
                                )
                            except Exception as e:
                                print(
                                    f"  Warning: Could not load data for client {client_id}: {e}"
                                )
                return

        os.makedirs(client_data_base_path, exist_ok=True)

        tokenized_dataset_path = os.path.join(
            self.config["data_base_path"],
            self.config["dataset_name"],
            "processed",
            "tokenized",
        )
        dataset = load_from_disk(tokenized_dataset_path)["train"]

        num_clients = self.config["num_clients"]
        indices = list(range(len(dataset)))
        base_seed = int(self.config.get("random_seed", 42))

        legacy_split = bool(self.config.get("legacy_client_split", False))
        legacy_seed = self.config.get("legacy_seed")
        if legacy_split:
            # Legacy split: round-robin in original order (no shuffle).
            pass
        else:
            # Default behavior: shuffle indices before round-robin.
            if legacy_seed is None:
                rng = random.Random(base_seed)
                rng.shuffle(indices)
            else:
                rng = random.Random(int(legacy_seed))
                rng.shuffle(indices)

        # Cria splits conforme a estratégia configurada
        strategy = self.config.get("data_distribution_strategy", "iid")
        client_splits = {i: [] for i in range(num_clients)}

        if legacy_split:
            eval_split = float(self.config.get("legacy_client_eval_split", 0.0))
            for i in range(num_clients):
                client_indices = indices[i::num_clients]
                eval_size = int(len(client_indices) * eval_split)
                client_splits[i] = client_indices[eval_size:]
        elif strategy == "iid":
            # Distribuição IID simples (round-robin após embaralhar)
            for i in range(num_clients):
                client_splits[i] = indices[i::num_clients]
        elif strategy == "quantity_skew_dirichlet":
            try:
                import numpy as np
            except ImportError:
                print("Warning: numpy not available, falling back to IID split.")
                for i in range(num_clients):
                    client_splits[i] = indices[i::num_clients]
            else:
                alpha = float(self.config.get("non_iid_alpha", 0.5))
                dirichlet = np.random.dirichlet([alpha] * num_clients)
                counts = (dirichlet * len(indices)).astype(int)
                # Ajusta para garantir que a soma dos counts seja exatamente o total
                diff = len(indices) - int(counts.sum())
                step = 1 if diff > 0 else -1
                for i in range(abs(diff)):
                    counts[i % num_clients] += step

                cursor = 0
                for client_id, count in enumerate(counts):
                    client_splits[client_id] = indices[cursor : cursor + int(count)]
                    cursor += int(count)
        elif strategy == "hetero_device":
            # Simula grupos de dispositivos com diferentes capacidades.
            # Pequeno (~20%), médio (~40%), grande (~40%) com pesos distintos.
            client_ids = list(range(num_clients))
            small_end = max(1, int(0.2 * num_clients))
            medium_end = max(small_end + 1, int(0.6 * num_clients))

            size_weights = {}
            for cid in client_ids[:small_end]:
                size_weights[cid] = 1.0  # dispositivos leves
            for cid in client_ids[small_end:medium_end]:
                size_weights[cid] = 3.0  # dispositivos intermediários
            for cid in client_ids[medium_end:]:
                size_weights[cid] = 6.0  # gateways/edge poderosos

            total_weight = sum(size_weights.values())
            clients = list(size_weights.keys())
            weights = [size_weights[cid] / total_weight for cid in clients]

            for idx in indices:
                chosen = random.choices(clients, weights=weights, k=1)[0]
                client_splits[chosen].append(idx)
        elif strategy == "by_src_ip":
            # Distribui por dispositivo (src_ip), criando um cenário Non-IID mais forte:
            # cada "dispositivo" (src_ip) fica inteiro em um único cliente.
            if "src_ip" not in dataset.column_names:
                print(
                    "Warning: 'src_ip' column not found in tokenized dataset. Falling back to IID split."
                )
                for i in range(num_clients):
                    client_splits[i] = indices[i::num_clients]
            else:
                src_ips = dataset["src_ip"]
                ip_to_indices = {}
                for idx in indices:
                    ip = src_ips[idx]
                    ip_to_indices.setdefault(ip, []).append(idx)

                unique_ips = list(ip_to_indices.keys())
                rng = random.Random(base_seed)
                rng.shuffle(unique_ips)

                for i, ip in enumerate(unique_ips):
                    client_id = i % num_clients
                    client_splits[client_id].extend(ip_to_indices[ip])
        else:
            print(
                f"Warning: Unknown data_distribution_strategy='{strategy}', falling back to IID."
            )
            for i in range(num_clients):
                client_splits[i] = indices[i::num_clients]

        # Salva shards e registra contagem de amostras por cliente
        self.client_sample_counts = {}
        for client_id, client_indices in client_splits.items():
            client_dataset = dataset.select(client_indices)
            self.client_sample_counts[client_id] = len(client_dataset)
            client_dataset.save_to_disk(
                os.path.join(client_data_base_path, f"client_{client_id}")
            )

        if metadata_path:
            try:
                with open(metadata_path, "w") as f:
                    json.dump(self.client_sample_counts, f)
            except Exception as e:
                print(f"Warning: Failed to save client metadata: {e}")

        print(
            f"Data successfully split for {num_clients} clients using strategy '{strategy}'."
        )

    def _get_learning_rate(self, current_round):
        """Calculates the learning rate for the current round based on the schedule."""
        if self.config["lr_scheduler_type"] == "cosine":
            initial_lr = self.config["initial_lr"]
            min_lr = self.config["min_lr"]
            total_rounds = self.config["num_rounds"]

            return min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * current_round / total_rounds)
            )
        else:  # constant
            return self.config["initial_lr"]

    def _get_adapters(self, model):
        """Extracts all trainable weights from a parameter-efficient model."""
        return {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _set_adapters(self, model, aggregated_adapters):
        """Updates the model with aggregated LoRA adapter weights."""
        for name, param in model.named_parameters():
            if name in aggregated_adapters:
                param.data.copy_(aggregated_adapters[name].to(self.device))

    def _aggregate_models(self, client_weights_list, client_ids=None):
        """
        Aggregates client model weights (adapters or full) using FedAvg.
        The aggregation is done on the CPU.
        """
        # The keys are the same for all clients, so we can take them from the first one.
        if not client_weights_list:
            print("Warning: Client weights list is empty. Skipping aggregation.")
            return

        aggregated_weights = {}
        weight_keys = client_weights_list[0].keys()

        # Configuração opcional: agregação ponderada pelo número de amostras por cliente
        use_weighted_agg = self.config.get("use_weighted_aggregation", False)
        sample_weights = None
        if use_weighted_agg and client_ids is not None and self.client_sample_counts:
            counts = [self.client_sample_counts.get(cid, 0) for cid in client_ids]
            total = sum(counts)
            if total > 0:
                sample_weights = [c / total for c in counts]
            else:
                print(
                    "Warning: Sample counts sum to zero. Falling back to unweighted aggregation."
                )
                sample_weights = None

        for key in weight_keys:
            # Stack all tensors for the current key from all clients and average them.
            # The tensors are on the CPU, so this uses RAM, not VRAM.
            if sample_weights is not None:
                stacked = torch.stack(
                    [
                        sample_weights[i] * client_weights_list[i][key]
                        for i in range(len(client_weights_list))
                    ],
                    dim=0,
                )
                aggregated_weights[key] = torch.sum(stacked, dim=0)
            else:
                aggregated_weights[key] = torch.mean(
                    torch.stack([weights[key] for weights in client_weights_list]),
                    dim=0,
                )

        if self.config["lora"]:
            self._set_adapters(self.global_model, aggregated_weights)
        else:
            # For full fine-tuning, update the entire model state dict
            # Move weights to GPU before loading them into the model
            gpu_aggregated_weights = {
                k: v.to(self.device) for k, v in aggregated_weights.items()
            }
            self.global_model.load_state_dict(gpu_aggregated_weights)

    def _abort_if_round_failed(
        self,
        round_num,
        selected_client_ids,
        successful_client_ids,
        failure_messages=None,
    ):
        if successful_client_ids:
            return

        details = ""
        if failure_messages:
            preview = " | ".join(str(msg) for msg in failure_messages[:3])
            details = f" Sample failures: {preview}"

        raise RuntimeError(
            f"Round {round_num} aborted: all selected clients failed "
            f"({selected_client_ids}). No aggregation was performed and continuing "
            f"would produce invalid checkpoints.{details}"
        )

    def _select_clients_for_round(self, round_num):
        """
        Selects clients for the current round according to the configured strategy.
        Supports uniform random selection and selection proportional to local data size.
        """
        num_clients = self.config["num_clients"]
        requested = int(num_clients * self.config["client_frac"])
        num_selected_clients = max(1, min(num_clients, requested))
        all_clients = list(range(num_clients))
        base_seed = int(self.config.get("random_seed", 42))

        strategy = self.config.get("client_selection_strategy", "uniform")

        if strategy == "data_size_proportional" and self.client_sample_counts:
            eligible = [
                (cid, self.client_sample_counts.get(cid, 0)) for cid in all_clients
            ]
            eligible = [(cid, c) for cid, c in eligible if c > 0]
            if not eligible:
                print(
                    "Warning: Invalid or zero client sample counts. Falling back to uniform selection."
                )
                return random.sample(all_clients, num_selected_clients)

            eligible_clients, eligible_counts = zip(*eligible)
            total = float(sum(eligible_counts))
            probabilities = [float(c) / total for c in eligible_counts]

            if num_selected_clients > len(eligible_clients):
                print(
                    "Warning: Requested more clients than those with data "
                    f"({num_selected_clients} > {len(eligible_clients)}). Capping selection."
                )
                num_selected_clients = len(eligible_clients)

            # Weighted sampling without replacement.
            rng = np.random.default_rng(seed=base_seed + int(round_num))
            selected_clients_ids = rng.choice(
                np.array(eligible_clients, dtype=int),
                size=int(num_selected_clients),
                replace=False,
                p=np.array(probabilities, dtype=float),
            ).tolist()
            print(
                f"Client selection strategy 'data_size_proportional' selected: {selected_clients_ids}"
            )
            return selected_clients_ids

        # Estratégia padrão: seleção uniforme aleatória
        if bool(self.config.get("legacy_client_selection_systemrandom", False)):
            legacy_seed = self.config.get("legacy_seed")
            if legacy_seed is None:
                rs = random.Random(base_seed + int(round_num))
                selected_clients_ids = rs.sample(all_clients, num_selected_clients)
            else:
                rng = random.Random(int(legacy_seed) + int(round_num))
                selected_clients_ids = rng.sample(all_clients, num_selected_clients)
        else:
            rng = random.Random(base_seed + int(round_num))
            selected_clients_ids = rng.sample(all_clients, num_selected_clients)
        print(f"Client selection strategy 'uniform' selected: {selected_clients_ids}")
        return selected_clients_ids

    def run_federated_training(self):
        """The main federated training loop."""
        self._split_data_for_clients()

        # --- Lógica Condicional para Paralelismo ---
        if self.config.get("use_parallel_training", False):
            self._run_parallel_training()
        else:
            self._run_sequential_training()

        self._save_communication_metrics()

    def _run_sequential_training(self):
        """Executes training sequentially in the main process."""
        print("--- Running in Sequential Mode ---")
        total_rounds = self.config["num_rounds"]
        round_times = []
        training_start = time.time()

        for round_num in range(1, total_rounds + 1):
            round_start = time.time()

            # ETA calculation
            if round_times:
                avg_round_time = sum(round_times) / len(round_times)
                eta_seconds = avg_round_time * (total_rounds - round_num + 1)
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "calculating..."

            print(f"\n{'=' * 60}")
            print(f"  Round {round_num}/{total_rounds} | ETA: {eta_str}")
            print(f"{'=' * 60}")

            selected_clients_ids = self._select_clients_for_round(round_num)

            client_weights_list = []
            successful_client_ids = []
            failure_messages = []
            current_lr = self._get_learning_rate(round_num)
            for client_id in selected_clients_ids:
                client_trainer = ClientTrainer(client_id, self.config)
                try:
                    cpu_weights = client_trainer.train(round_num, current_lr)
                    if cpu_weights:
                        client_weights_list.append(cpu_weights)
                        successful_client_ids.append(client_id)
                except Exception as e:
                    msg = f"client_{client_id}: {e}"
                    failure_messages.append(msg)
                    print(f"Erro ao treinar cliente: {msg}")

            self._abort_if_round_failed(
                round_num,
                selected_clients_ids,
                successful_client_ids,
                failure_messages,
            )

            print("Aggregating client models...")
            self._aggregate_models(client_weights_list, successful_client_ids)

            # Communication metrics (bytes communicated this round)
            self._record_round_communication(
                round_num, successful_client_ids, client_weights_list
            )

            round_model_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                f"round_{round_num}",
                "global_model",
            )
            os.makedirs(round_model_path, exist_ok=True)
            self.global_model.save_pretrained(round_model_path)

            # Time tracking
            round_elapsed = time.time() - round_start
            round_times.append(round_elapsed)
            total_elapsed = time.time() - training_start

            print(
                f"Round {round_num} completed in {format_time(round_elapsed)} | Total: {format_time(total_elapsed)}"
            )

        # Final summary
        total_time = time.time() - training_start
        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Avg per round: {format_time(total_time / total_rounds)}")
        print(f"{'=' * 60}")

    def _run_parallel_training(self):
        """Executes training in parallel across multiple GPUs."""
        print("--- Running in Parallel Mode ---")
        available_gpus = torch.cuda.device_count()
        configured_max_gpus = int(self.config.get("max_parallel_gpus", available_gpus))
        num_gpus = min(available_gpus, configured_max_gpus)
        if num_gpus == 0:
            print("AVISO: Nenhuma GPU encontrada. Voltando para o modo sequencial.")
            self._run_sequential_training()
            return

        print(f"Encontradas {num_gpus} GPUs. Distribuindo clientes entre elas.")

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        total_rounds = self.config["num_rounds"]
        round_times = []
        training_start = time.time()

        for round_num in range(1, total_rounds + 1):
            round_start = time.time()

            # ETA calculation
            if round_times:
                avg_round_time = sum(round_times) / len(round_times)
                eta_seconds = avg_round_time * (total_rounds - round_num + 1)
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "calculating..."

            print(f"\n{'=' * 60}")
            print(f"  Round {round_num}/{total_rounds} | ETA: {eta_str}")
            print(f"{'=' * 60}")

            selected_clients_ids = self._select_clients_for_round(round_num)

            client_weights_list = []
            successful_client_ids = []
            failure_messages = []
            current_lr = self._get_learning_rate(round_num)

            # Mover o modelo global para a CPU para liberar VRAM para os clientes
            self.global_model.to("cpu")
            torch.cuda.empty_cache()

            client_chunks = list(chunks(selected_clients_ids, num_gpus))
            for i, client_chunk in enumerate(client_chunks):
                print(f"  --- Processing client batch {i + 1}/{len(client_chunks)} ---")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_gpus
                ) as executor:
                    future_to_client = {}
                    for j, client_id in enumerate(client_chunk):
                        gpu_id = j % num_gpus
                        args = (client_id, self.config, round_num, current_lr, gpu_id)
                        future = executor.submit(train_client_process, args)
                        future_to_client[future] = client_id

                    for future in concurrent.futures.as_completed(future_to_client):
                        client_id = future_to_client[future]
                        try:
                            cpu_weights = future.result()
                            if cpu_weights:
                                client_weights_list.append(cpu_weights)
                                successful_client_ids.append(client_id)
                        except Exception as e:
                            msg = f"client_{client_id}: {e}"
                            failure_messages.append(msg)
                            print(f"Erro ao treinar cliente: {msg}")

            # Mover o modelo de volta para a GPU para agregação
            self.global_model.to(self.device)

            self._abort_if_round_failed(
                round_num,
                selected_clients_ids,
                successful_client_ids,
                failure_messages,
            )

            print("Aggregating client models...")
            self._aggregate_models(client_weights_list, successful_client_ids)

            # Communication metrics (bytes communicated this round)
            self._record_round_communication(
                round_num, successful_client_ids, client_weights_list
            )

            round_model_path = os.path.join(
                self.config["results_path"],
                self.config["simulation_name"],
                f"round_{round_num}",
                "global_model",
            )
            os.makedirs(round_model_path, exist_ok=True)
            self.global_model.save_pretrained(round_model_path)

            # Time tracking
            round_elapsed = time.time() - round_start
            round_times.append(round_elapsed)
            total_elapsed = time.time() - training_start

            print(
                f"Round {round_num} completed in {format_time(round_elapsed)} | Total: {format_time(total_elapsed)}"
            )

        # Final summary
        total_time = time.time() - training_start
        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Avg per round: {format_time(total_time / total_rounds)}")
        print(f"{'=' * 60}")
