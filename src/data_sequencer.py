import pandas as pd
import json
import logging
from typing import Dict, Any

# Standardized logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads configuration parameters from a JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def generate_sequences() -> None:
    """
    Reads the cleaned data, sorts it chronologically, and extracts sequential
    interaction histories for deep learning models.
    """
    try:
        config = load_config()
        cleaned_file = config["output_file"]
        sequence_file = config["sequence_file"]
        max_seq_len = config["max_sequence_length"]

        logging.info("Starting Phase 1: Sequential Data Generation...")
        logging.info(
            f"Loading '{cleaned_file}'. This requires significant RAM, please wait..."
        )

        # Load the cleaned data. We specify columns to save memory.
        cols_to_use = ["user_id", "item_id", "item_category", "behavior_type", "time"]
        df = pd.read_csv(cleaned_file, usecols=cols_to_use)

        logging.info("Sorting data strictly by chronological time...")
        df.sort_values(by=["user_id", "time"], inplace=True)

        logging.info("Grouping interactions into time-series sequences...")
        # We aggregate the historical actions into Python lists
        sequences = (
            df.groupby("user_id")
            .agg({"item_id": list, "item_category": list, "behavior_type": list})
            .reset_index()
        )

        logging.info(
            f"Truncating/Padding sequences to a maximum length of {max_seq_len}..."
        )

        # Function to safely slice the most recent N interactions
        def pad_or_truncate(seq_list, max_len):
            if len(seq_list) >= max_len:
                return seq_list[-max_len:]  # Take the most recent interactions
            else:
                # Pad with 0s if the user has a short history
                return [0] * (max_len - len(seq_list)) + seq_list

        sequences["item_seq"] = sequences["item_id"].apply(
            lambda x: pad_or_truncate(x, max_seq_len)
        )
        sequences["category_seq"] = sequences["item_category"].apply(
            lambda x: pad_or_truncate(x, max_seq_len)
        )
        sequences["behavior_seq"] = sequences["behavior_type"].apply(
            lambda x: pad_or_truncate(x, max_seq_len)
        )

        # Drop the raw lists to save memory
        sequences.drop(
            columns=["item_id", "item_category", "behavior_type"], inplace=True
        )

        logging.info(f"Saving sequences to Pandas Pickle format: '{sequence_file}'...")
        sequences.to_pickle(sequence_file)

        logging.info("Success! Data is now perfectly shaped for PyTorch.")
        logging.info(f"Total Users processed: {len(sequences)}")

    except MemoryError:
        logging.error(
            "Your Mac ran out of RAM. We will need to process this in smaller chunks."
        )
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    generate_sequences()
