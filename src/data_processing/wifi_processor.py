import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .base_processor import BaseProcessor
from src.utils.hf import hf_from_pretrained_kwargs


class WiFiProcessor(BaseProcessor):
    """
    Processor for Wi-Fi / network-flow CSV datasets.

    Supported input mode:
      - a single raw CSV placed in data/wifi/raw/
      - labels in a column such as "Label"
    """

    DEFAULT_CONTENT_COLUMNS = [
        "Protocol",
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Byts/s",
        "Flow Bytes/s",
        "Flow Pkts/s",
        "Flow IAT Mean",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Pkt Len Mean",
        "Pkt Len Var",
        "SYN Flag Cnt",
        "ACK Flag Cnt",
        "PSH Flag Cnt",
    ]

    METADATA_COLUMNS = [
        "Flow ID",
        "Src IP",
        "Src Port",
        "Dst IP",
        "Dst Port",
        "Protocol",
        "Timestamp",
        "Label_raw",
        "Label",
        "Content",
    ]

    def _get_training_task(self) -> str:
        return str(self.config.get("training_task", "causal_lm")).strip().lower()

    def _get_split_strategy(self) -> str:
        return str(
            self.config.get("wifi_split_strategy", "benign_only_anomaly_detection")
        ).strip().lower()

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        return slug or "field"

    def _column_lookup(self, columns: Iterable[str]) -> Dict[str, str]:
        return {self._normalize_key(col): col for col in columns}

    def _resolve_existing_columns(
        self, df: pd.DataFrame, candidates: Iterable[str]
    ) -> List[str]:
        lookup = self._column_lookup(df.columns)
        resolved: List[str] = []
        seen = set()
        for candidate in candidates:
            actual = lookup.get(self._normalize_key(candidate))
            if actual and actual not in seen:
                resolved.append(actual)
                seen.add(actual)
        return resolved

    def _format_value(self, value) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                return "INF" if value > 0 else "NEG_INF"
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.6g}"
        return str(value).strip()

    def _read_csv(self, file_path: str) -> pd.DataFrame:
        read_kwargs = {
            "skipinitialspace": True,
            "low_memory": False,
        }
        raw_nrows = self.config.get("raw_nrows")
        if raw_nrows is not None:
            read_kwargs["nrows"] = int(raw_nrows)
        return pd.read_csv(file_path, **read_kwargs)

    def _resolve_raw_csv_path(self) -> str:
        configured_name = self.config.get("raw_csv_file")
        if configured_name:
            candidate = (
                configured_name
                if os.path.isabs(configured_name)
                else os.path.join(self.raw_path, configured_name)
            )
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"Configured raw CSV not found: {candidate}. "
                    f"Place the dataset in {self.raw_path} or update raw_csv_file."
                )
            return candidate

        csv_candidates = sorted(
            file_name
            for file_name in os.listdir(self.raw_path)
            if file_name.lower().endswith(".csv")
        )
        if len(csv_candidates) == 1:
            return os.path.join(self.raw_path, csv_candidates[0])
        if len(csv_candidates) > 1:
            raise ValueError(
                "Multiple raw CSV files found in wifi/raw and raw_csv_file is not set. "
                f"Candidates: {csv_candidates}"
            )
        return ""

    def _detect_label_column(self, df: pd.DataFrame) -> str:
        configured = self.config.get("label_column", "Label")
        resolved = self._resolve_existing_columns(df, [configured, "label", "Label"])
        if not resolved:
            raise ValueError(
                f"No label column found in raw CSV. Expected something like '{configured}'."
            )
        return resolved[0]

    def _normalize_label_series(self, series: pd.Series) -> pd.DataFrame:
        raw_values = series.where(series.notna(), other="").astype(str).str.strip()
        numeric = pd.to_numeric(raw_values, errors="coerce")

        numeric_non_null = numeric.dropna()
        if not numeric_non_null.empty and set(numeric_non_null.astype(int).unique()).issubset({0, 1}):
            label = numeric.astype("Int64")
            raw_label = raw_values
            return pd.DataFrame({"Label_raw": raw_label, "Label": label})

        configured_normal = self.config.get(
            "normal_label_values", ["Benign", "BENIGN", "benign", "Normal", "normal"]
        )
        normal_values = {self._normalize_key(v) for v in configured_normal}
        normal_values.update({"0", "benign", "normal"})

        raw_label = raw_values
        binary = raw_label.apply(
            lambda value: (
                np.nan
                if value == ""
                else 0
                if self._normalize_key(value) in normal_values
                else 1
            )
        )
        return pd.DataFrame({"Label_raw": raw_label, "Label": binary})

    def _load_single_raw_dataframe(self) -> pd.DataFrame:
        raw_csv_path = self._resolve_raw_csv_path()
        if not raw_csv_path:
            raise FileNotFoundError(
                f"No supported raw WiFi dataset found in {self.raw_path}. "
                "Expected the CIC-IDS2018 raw CSV configured by raw_csv_file."
            )

        print(f"Loading WiFi flows from: {raw_csv_path}")
        df = self._read_csv(raw_csv_path)
        df.columns = df.columns.str.strip()

        label_col = self._detect_label_column(df)
        label_df = self._normalize_label_series(df[label_col])
        df = df.drop(columns=[label_col]).copy()
        df["Label_raw"] = label_df["Label_raw"]
        df["Label"] = label_df["Label"]
        df = df.dropna(subset=["Label"]).copy()
        df["Label"] = df["Label"].astype(int)

        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                df.loc[df[col].isin(["", "nan", "None", "NaN"]), col] = np.nan
        return df

    def _select_content_columns(self, df: pd.DataFrame) -> List[str]:
        configured = self.config.get("wifi_content_columns", self.DEFAULT_CONTENT_COLUMNS)
        selected = self._resolve_existing_columns(df, configured)
        if selected:
            return selected

        excluded = {
            self._normalize_key(name)
            for name in [
                "Label",
                "Label_raw",
                "Flow ID",
                "Src IP",
                "Dst IP",
                "Timestamp",
            ]
        }
        fallback = []
        for column in df.columns:
            if self._normalize_key(column) in excluded:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                fallback.append(column)
        return fallback[:12]

    def _build_text_from_row(self, row: pd.Series, content_columns: List[str]) -> str:
        parts = []
        for column in content_columns:
            value = self._format_value(row.get(column))
            parts.append(f"{self._slugify(column)} {value}")
        return " ".join(parts)

    def _add_content_column(
        self, df: pd.DataFrame, content_columns: List[str]
    ) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            df["Content"] = pd.Series(dtype="object")
            return df
        df["Content"] = df.apply(
            lambda row: self._build_text_from_row(row, content_columns), axis=1
        )
        return df

    def _finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered = [col for col in self.METADATA_COLUMNS if col in df.columns]
        remaining = [col for col in df.columns if col not in ordered]
        return df[ordered + remaining].copy()

    def _balance_test_set(self, df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
        if not self.config.get("test_balance", False):
            return df

        benign_df = df[df["Label"] == 0]
        anomaly_df = df[df["Label"] == 1]
        if benign_df.empty or anomaly_df.empty:
            return df

        sample_size = min(len(benign_df), len(anomaly_df))
        benign_sample = benign_df.sample(n=sample_size, random_state=seed)
        anomaly_sample = anomaly_df.sample(n=sample_size, random_state=seed)
        return (
            pd.concat([benign_sample, anomaly_sample], ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

    def _cap_dataframe(
        self, df: pd.DataFrame, cap: Optional[int], *, seed: int, label: str
    ) -> pd.DataFrame:
        if cap is None:
            return df
        cap = int(cap)
        if cap <= 0:
            return df.iloc[0:0].copy()
        if len(df) <= cap:
            return df

        print(f"Capping {label}: {len(df)} -> {cap} rows")
        return df.sample(n=cap, random_state=seed).reset_index(drop=True)

    def _balanced_binary_split(
        self, benign_df: pd.DataFrame, anomaly_df: pd.DataFrame, *, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        if self._get_training_task() != "sequence_classification":
            raise ValueError(
                "wifi_split_strategy='balanced_binary' requires training_task='sequence_classification'."
            )

        anchor_mode = str(
            self.config.get("balanced_anchor", "minority")
        ).strip().lower()
        explicit_target = self.config.get("balanced_class_cap")

        if explicit_target is not None:
            target_per_class = min(
                int(explicit_target), len(benign_df), len(anomaly_df)
            )
        elif anchor_mode == "minority":
            target_per_class = min(len(benign_df), len(anomaly_df))
        else:
            raise ValueError(
                "Unsupported balanced_anchor. Use 'minority' or set balanced_class_cap."
            )

        if target_per_class <= 1:
            raise ValueError("Balanced binary split produced too few samples per class.")

        benign_target = benign_df.iloc[:target_per_class].copy()
        anomaly_target = anomaly_df.iloc[:target_per_class].copy()
        benign_surplus = benign_df.iloc[target_per_class:].copy()

        train_frac = float(self.config.get("train_fraction_per_class", 0.8))
        train_frac = min(max(train_frac, 0.0), 1.0)
        train_len = int(target_per_class * train_frac)
        train_len = min(max(train_len, 1), target_per_class - 1)

        benign_train = benign_target.iloc[:train_len].copy()
        benign_test = benign_target.iloc[train_len:].copy()
        anomaly_train = anomaly_target.iloc[:train_len].copy()
        anomaly_test = anomaly_target.iloc[train_len:].copy()

        if benign_train.empty or anomaly_train.empty:
            raise ValueError(
                "Balanced training split is empty for at least one class. "
                "Increase train_fraction_per_class or class cap."
            )
        if benign_test.empty or anomaly_test.empty:
            raise ValueError(
                "Balanced test split is empty for at least one class. "
                "Reduce train_fraction_per_class or increase class cap."
            )

        train_df = (
            pd.concat([benign_train, anomaly_train], ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        test_df = (
            pd.concat([benign_test, anomaly_test], ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

        calibration_df = None
        calibration_cap = self.config.get("calibration_benign_cap")
        calibration_cap = int(calibration_cap) if calibration_cap else 0
        if calibration_cap > 0 and not benign_surplus.empty:
            calibration_df = self._cap_dataframe(
                benign_surplus,
                calibration_cap,
                seed=seed,
                label="balanced_binary_calibration_benign",
            )

        print(
            "Balanced binary WiFi split selected "
            f"{target_per_class} benign + {target_per_class} anomaly rows "
            f"(total={target_per_class * 2})."
        )
        if not benign_surplus.empty:
            print(f"Remaining benign surplus after class matching: {len(benign_surplus)}")

        return train_df, test_df, calibration_df

    def create_sessions(self):
        print("Creating WiFi train/test splits from raw network-flow data...")
        df = self._clean_dataframe(self._load_single_raw_dataframe())

        benign_df = df[df["Label"] == 0].copy()
        anomaly_df = df[df["Label"] == 1].copy()
        if benign_df.empty:
            raise ValueError("No benign rows found after label normalization.")
        if anomaly_df.empty:
            raise ValueError("No anomalous rows found after label normalization.")

        content_columns = self._select_content_columns(df)
        if not content_columns:
            raise ValueError(
                "No usable feature columns found to build the Content field. "
                "Check wifi_content_columns or the raw CSV schema."
            )
        print(f"Using {len(content_columns)} feature columns for Content: {content_columns}")

        seed = int(self.config.get("random_seed", 42))
        benign_df = benign_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        anomaly_df = anomaly_df.sample(frac=1.0, random_state=seed).reset_index(
            drop=True
        )

        split_strategy = self._get_split_strategy()
        calibration_df: Optional[pd.DataFrame] = None

        if split_strategy == "balanced_binary":
            train_df, test_df, calibration_df = self._balanced_binary_split(
                benign_df, anomaly_df, seed=seed
            )
            train_df = self._add_content_column(train_df, content_columns)
            test_df = self._add_content_column(test_df, content_columns)
            if calibration_df is not None and not calibration_df.empty:
                calibration_df = self._add_content_column(calibration_df, content_columns)
        else:
            train_frac = float(self.config.get("train_benign_fraction", 0.8))
            train_frac = min(max(train_frac, 0.0), 1.0)
            calib_frac = float(self.config.get("benign_calibration_fraction", 0.0))
            calib_frac = min(max(calib_frac, 0.0), 1.0)
            benign_train_cap = self.config.get("benign_train_cap")
            benign_train_cap = int(benign_train_cap) if benign_train_cap else None

            if benign_train_cap is not None:
                train_len = min(benign_train_cap, len(benign_df))
            else:
                train_len = int(len(benign_df) * train_frac)

            remaining_after_train = max(0, len(benign_df) - train_len)
            calib_len = int(remaining_after_train * calib_frac)

            benign_train = benign_df.iloc[:train_len].copy()
            benign_calibration = benign_df.iloc[train_len : train_len + calib_len].copy()
            benign_holdout = benign_df.iloc[train_len + calib_len :].copy()
            if benign_train.empty:
                raise ValueError(
                    "Training split is empty. Increase train_benign_fraction or benign_train_cap."
                )

            benign_train = self._cap_dataframe(
                benign_train,
                self.config.get("benign_train_cap"),
                seed=seed,
                label="benign_train",
            )
            benign_calibration = self._cap_dataframe(
                benign_calibration,
                self.config.get("calibration_benign_cap"),
                seed=seed,
                label="benign_calibration",
            )
            benign_holdout = self._cap_dataframe(
                benign_holdout,
                self.config.get("test_benign_cap"),
                seed=seed,
                label="test_benign_holdout",
            )
            anomaly_df = self._cap_dataframe(
                anomaly_df,
                self.config.get("test_anomaly_cap"),
                seed=seed,
                label="test_anomaly",
            )

            benign_train = self._add_content_column(benign_train, content_columns)
            benign_calibration = self._add_content_column(
                benign_calibration, content_columns
            )
            benign_holdout = self._add_content_column(benign_holdout, content_columns)
            anomaly_df = self._add_content_column(anomaly_df, content_columns)

            train_df = benign_train
            test_df = self._balance_test_set(
                pd.concat([benign_holdout, anomaly_df], ignore_index=True)
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True),
                seed=seed,
            )
            calibration_df = (
                benign_calibration if not benign_calibration.empty else None
            )

        train_df = self._finalize_columns(train_df)
        test_df = self._finalize_columns(test_df)
        train_df.to_csv(os.path.join(self.processed_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test.csv"), index=False)

        if calibration_df is not None and not calibration_df.empty:
            calibration_df = self._finalize_columns(calibration_df)
            calibration_df.to_csv(
                os.path.join(self.processed_path, "calibration.csv"), index=False
            )

        print(
            f"Created train.csv ({len(train_df)} rows; benign={int((train_df['Label'] == 0).sum())}, "
            f"anomaly={int((train_df['Label'] == 1).sum())}) and test.csv "
            f"({len(test_df)} rows; benign={int((test_df['Label'] == 0).sum())}, "
            f"anomaly={int((test_df['Label'] == 1).sum())}) in {self.processed_path}"
        )
        if calibration_df is not None and not calibration_df.empty:
            print(
                f"Created calibration.csv with {len(calibration_df)} benign rows "
                f"in {self.processed_path}"
            )

    def preprocess_and_sanitize(self):
        """
        Sanitizes textual content while preserving the flow metrics that matter
        for anomaly detection.
        """
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        mac_pattern = r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"

        def apply_regex(text: str) -> str:
            text = str(text)
            text = re.sub(ip_pattern, "IP_ADDR", text)
            text = re.sub(mac_pattern, "MAC_ADDR", text)
            return text

        print("Sanitizing WiFi processed splits...")
        for split in ["train", "test", "calibration"]:
            file_path = os.path.join(self.processed_path, f"{split}.csv")
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df["Content"] = df["Content"].astype(str).apply(apply_regex)
            df.to_csv(file_path, index=False)
        print("Sanitization complete.")

    def tokenize_dataset(self):
        print("Tokenizing WiFi dataset...")

        train_path = os.path.join(self.processed_path, "train.csv")
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            print(f"Error: {train_path} not found. Did create_sessions run correctly?")
            return

        training_task = self._get_training_task()
        if training_task == "sequence_classification":
            if "Label" not in df.columns:
                print("Error: train.csv does not contain the Label column.")
                return
            dataset_source = df.copy()
            dataset_columns = ["Content", "Label"]
            rename_map = {"Content": "text", "Label": "labels"}
        else:
            dataset_source = df[df["Label"] == 0].copy()
            if dataset_source.empty:
                print("Warning: No normal data (Label == 0) found for training.")
                return
            dataset_columns = ["Content"]
            rename_map = {"Content": "text"}

        if "Src IP" in dataset_source.columns:
            dataset_columns.append("Src IP")
            rename_map["Src IP"] = "src_ip"
        if "Timestamp" in dataset_source.columns:
            dataset_columns.append("Timestamp")
            rename_map["Timestamp"] = "timestamp"

        dataset = Dataset.from_pandas(
            dataset_source[dataset_columns].rename(columns=rename_map),
            preserve_index=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], **hf_from_pretrained_kwargs(self.config)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        def preprocess(examples):
            max_len = int(
                self.config.get("max_length", self.config.get("eval_max_length", 1024))
            )
            if training_task == "sequence_classification":
                return tokenizer(examples["text"], truncation=True, max_length=max_len)

            examples["text"] = [
                text + tokenizer.eos_token if tokenizer.eos_token else text
                for text in examples["text"]
            ]
            use_legacy = bool(
                self.config.get("use_legacy_tokenization", False)
                or self.config.get("use_legacy_trainer", False)
            )
            if use_legacy:
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    padding_side="right",
                )
            return tokenizer(examples["text"], truncation=True, max_length=max_len)

        tokenized = dataset.map(preprocess, batched=True, remove_columns=["text"])

        if training_task != "sequence_classification" and self.config.get(
            "use_legacy_trainer", False
        ):
            tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)

        final_dataset = DatasetDict({"train": tokenized})
        final_dataset.save_to_disk(self.tokenized_path)

        print(f"Tokenized WiFi dataset saved to {self.tokenized_path}")
