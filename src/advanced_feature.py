import pandas as pd
import json
import logging
from typing import Dict, Any, List

# Standardized logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Loads configuration parameters from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def pivot_and_rename(df: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
    """
    Helper function to aggregate counts and pivot behavior types into distinct columns.
    Ensures Excel compatibility by stripping the hidden MultiIndex name.
    """
    df_grouped = df.groupby(index_cols + ['behavior_type'])['count'].sum().reset_index()
    df_pivot = df_grouped.pivot(index=index_cols, columns='behavior_type', values='count').fillna(0)

    for col in [1, 2, 3, 4]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0

    df_pivot.rename(columns={1: 'pv_count', 2: 'fav_count', 3: 'cart_count', 4: 'buy_count'}, inplace=True)

    df_pivot['total_actions'] = df_pivot['pv_count'] + df_pivot['fav_count'] + df_pivot['cart_count'] + df_pivot[
        'buy_count']
    df_pivot['buy_rate'] = df_pivot['buy_count'] / df_pivot['total_actions'].replace(0, 1)

    df_pivot.columns.name = None
    return df_pivot.reset_index()


def extract_advanced_features(config_path: str = 'config.json') -> None:
    """
    Extracts the 5th (User-Category) and 6th (Temporal Trends) dimensional features.
    """
    config = load_config(config_path)
    cleaned_file = config['output_file']
    chunk_size = config['chunk_size']

    user_category_chunks: List[pd.DataFrame] = []
    time_chunks: List[pd.DataFrame] = []

    logging.info(f"Starting advanced feature extraction from '{cleaned_file}'...")

    try:
        for chunk in pd.read_csv(cleaned_file, chunksize=chunk_size):
            # --- PREPARATION FOR TIME DIMENSION ---
            # Convert time string back to datetime to extract the specific hour (0-23)
            chunk['hour'] = pd.to_datetime(chunk['time']).dt.hour

            # --- 5. User-Category Dimension ---
            uc_agg = chunk.groupby(['user_id', 'item_category', 'behavior_type']).size().reset_index(name='count')
            user_category_chunks.append(uc_agg)

            # --- 6. Temporal Dimension (By Hour) ---
            time_agg = chunk.groupby(['hour', 'behavior_type']).size().reset_index(name='count')
            time_chunks.append(time_agg)

            logging.info("Processed intermediate advanced chunk...")

        logging.info("All chunks read. Performing final aggregations...")

        # Process and save User-Category Features
        user_category_features = pivot_and_rename(pd.concat(user_category_chunks), ['user_id', 'item_category'])
        user_category_features.to_csv(config['user_category_file'], index=False, encoding='utf-8-sig')
        logging.info(f"User-Category features saved to '{config['user_category_file']}'")

        # Process and save Temporal Features
        time_features = pivot_and_rename(pd.concat(time_chunks), ['hour'])
        time_features.to_csv(config['time_features_file'], index=False, encoding='utf-8-sig')
        logging.info(f"Temporal features saved to '{config['time_features_file']}'")

        logging.info("Advanced Feature Engineering complete! All 6 dimensions are now ready.")

    except Exception as e:
        logging.error(f"An error occurred during advanced feature extraction: {e}")
        raise


if __name__ == "__main__":
    extract_advanced_features()