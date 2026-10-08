from abc import ABC, abstractmethod
import os

class BaseProcessor(ABC):
    """
    Abstract Base Class for data processing.
    Ensures that any new dataset processor implements a consistent interface.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.raw_path = os.path.join(config['data_base_path'], self.dataset_name, 'raw')
        self.processed_path = os.path.join(config['data_base_path'], self.dataset_name, 'processed')
        self.tokenized_path = os.path.join(self.processed_path, 'tokenized')

        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

    @abstractmethod
    def create_sessions(self):
        """
        Reads raw log files, groups them into sessions (e.g., by time, process),
        and saves them to train.csv and test.csv in the processed directory.
        """
        pass

    @abstractmethod
    def preprocess_and_sanitize(self):
        """
        Applies regex or other rules to sanitize the content of train.csv and test.csv.
        """
        pass

    @abstractmethod
    def tokenize_dataset(self):
        """
        Tokenizes the preprocessed data using the specified model's tokenizer
        and saves the tokenized dataset.
        """
        pass

    def run(self):
        """
        Executes the full data processing pipeline.
        """
        print(f"--- Starting data processing for dataset: {self.dataset_name} ---")
        required = self.config.get("processed_required_files", ["train.csv", "test.csv"])
        required_paths = [os.path.join(self.processed_path, f) for f in required]
        missing_required = any(not os.path.exists(p) for p in required_paths)

        if self.config.get('force_reprocess_data', False) or missing_required:
            print("Step 1: Creating sessions...")
            self.create_sessions()
            print("Step 2: Sanitizing data...")
            self.preprocess_and_sanitize()
        else:
            print("Skipping session creation and sanitization as processed files already exist.")

        if self.config.get('force_reprocess_data', False) or not os.path.exists(self.tokenized_path):
            print("Step 3: Tokenizing dataset...")
            self.tokenize_dataset()
        else:
            print("Skipping tokenization as tokenized dataset already exists.")
        print("--- Data processing complete. ---")
