import pandas as pd
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str = 'config.json') -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def generate_master_table():
    try:
        config = load_config()
        logging.info("Starting Stage 3: Data Merging and Label Generation...")

        # 1. Load the Base Table (User-Item Interactions)
        logging.info("Loading base user-item features...")
        master_df = pd.read_csv(config['user_item_features_file'])

        # 2. Generate the Target Label (Highest Intent Rule)
        logging.info("Calculating Target Labels...")
        # Using numpy select for blazing fast conditional logic
        conditions = [
            (master_df['buy_count'] > 0),
            (master_df['cart_count'] > 0),
            (master_df['fav_count'] > 0)
        ]
        choices = [4, 3, 2]
        master_df['target_label'] = np.select(conditions, choices, default=1)

        # 3. Merge User Dimensions
        logging.info("Merging User dimensions...")
        user_df = pd.read_csv(config['user_features_file'])
        # Add a suffix to avoid column name collisions (e.g., pv_count -> pv_count_user)
        master_df = master_df.merge(user_df, on='user_id', how='left', suffixes=('', '_user'))

        # 4. Merge Item Dimensions
        logging.info("Merging Item dimensions...")
        item_df = pd.read_csv(config['item_features_file'])
        master_df = master_df.merge(item_df, on='item_id', how='left', suffixes=('', '_item'))

        # 5. Clean up and Save
        master_file_name = '../data/processed/master_training_data.csv'
        master_df.fillna(0, inplace=True)  # Fill any missing merged data with 0
        master_df.to_csv(master_file_name, index=False, encoding='utf-8-sig')

        logging.info(f"Stage 3 Complete! Master table saved as '{master_file_name}'.")
        logging.info(f"Total rows ready for machine learning: {len(master_df)}")

    except Exception as e:
        logging.error(f"An error occurred during merging: {e}")
        raise


if __name__ == "__main__":
    generate_master_table()