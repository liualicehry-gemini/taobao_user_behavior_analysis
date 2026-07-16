import pandas as pd
import json
import logging
from typing import List, Dict, Any

# Set up standard logging instead of basic print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration parameters from a JSON file.

    Args:
        config_path (str): The file path to the configuration JSON.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        logging.info(f"Successfully loaded configuration from '{config_path}'")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{config_path}'")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{config_path}'")
        raise


def process_chunk(chunk: pd.DataFrame, valid_behaviors: List[int]) -> pd.DataFrame:
    """
    Cleans and optimizes a single chunk of e-commerce behavior data.

    Args:
        chunk (pd.DataFrame): The raw data chunk.
        valid_behaviors (List[int]): A list of acceptable integer behavior types.

    Returns:
        pd.DataFrame: The cleaned and memory-optimized data chunk.
    """
    # 1. Remove rows with missing crucial data
    chunk.dropna(subset=['user_id', 'behavior_type', 'time'], inplace=True)

    # 2. Filter for strictly valid behavior types
    chunk = chunk[chunk['behavior_type'].isin(valid_behaviors)]

    # 3. Parse Time and handle invalid calendar dates automatically
    chunk['time'] = pd.to_datetime(chunk['time'], format='mixed', errors='coerce')
    chunk.dropna(subset=['time'], inplace=True)

    # 4. Memory Optimization via Downcasting
    chunk['behavior_type'] = chunk['behavior_type'].astype('int8')
    chunk['user_id'] = pd.to_numeric(chunk['user_id'], downcast='integer')
    chunk['item_id'] = pd.to_numeric(chunk['item_id'], downcast='integer')
    chunk['item_category'] = pd.to_numeric(chunk['item_category'], downcast='integer')

    return chunk


def clean_data(config_path: str = 'config.json') -> None:
    """
    Main pipeline function to orchestrate reading, cleaning, and saving the dataset.

    Args:
        config_path (str): Path to the configuration file. Defaults to 'config.json'.
    """
    try:
        config = load_config(config_path)

        input_file = config['input_file']
        output_file = config['output_file']
        chunk_size = config['chunk_size']
        valid_behaviors = config['valid_behaviors']

        first_chunk = True
        total_rows_processed = 0

        logging.info(f"Starting data cleaning process for '{input_file}'...")

        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            cleaned_chunk = process_chunk(chunk, valid_behaviors)

            # Save the cleaned chunk incrementally
            if first_chunk:
                cleaned_chunk.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                cleaned_chunk.to_csv(output_file, index=False, mode='a', header=False)

            total_rows_processed += len(cleaned_chunk)
            logging.info(f"Processed a chunk. Total clean rows so far: {total_rows_processed}")

        logging.info(f"Data cleaning successfully complete! Cleaned file saved as '{output_file}'")

    except Exception as e:
        logging.error(f"A critical error occurred during data processing: {e}")
        raise


if __name__ == "__main__":
    clean_data()