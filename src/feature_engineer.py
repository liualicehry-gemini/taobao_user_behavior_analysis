import pandas as pd
import json
import logging
from typing import Dict, Any, List

# Standardized logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str) -> Dict[str, Any]:
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
    """
    # Final aggregation across all chunks
    df_grouped = df.groupby(index_cols + ['behavior_type'])['count'].sum().reset_index()

    # Pivot so each behavior type has its own column
    df_pivot = df_grouped.pivot(index=index_cols, columns='behavior_type', values='count').fillna(0)

    # Ensure all 4 behavior columns exist
    for col in [1, 2, 3, 4]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0

    # Rename for readability
    df_pivot.rename(columns={1: 'pv_count', 2: 'fav_count', 3: 'cart_count', 4: 'buy_count'}, inplace=True)

    # Calculate Conversion Features
    df_pivot['total_actions'] = df_pivot['pv_count'] + df_pivot['fav_count'] + df_pivot['cart_count'] + df_pivot[
        'buy_count']
    df_pivot['buy_rate'] = df_pivot['buy_count'] / df_pivot['total_actions'].replace(0, 1)

    # --- THE FIX: Remove the hidden MultiIndex name ---
    df_pivot.columns.name = None

    return df_pivot.reset_index()


def extract_features(config_path: str = 'config.json') -> None:
    """
    Reads cleaned data in chunks, extracts 4 dimensions, calculates behavioral ratios,
    and saves them to independent CSV files.
    """
    config = load_config(config_path)
    cleaned_file = config['output_file']
    chunk_size = config['chunk_size']

    # Lists to hold intermediate data
    user_chunks: List[pd.DataFrame] = []
    item_chunks: List[pd.DataFrame] = []
    category_chunks: List[pd.DataFrame] = []
    interaction_chunks: List[pd.DataFrame] = []

    logging.info(f"Starting 4-dimensional feature extraction from '{cleaned_file}'...")

    try:
        for chunk in pd.read_csv(cleaned_file, chunksize=chunk_size):
            # 1. User Dimension
            user_agg = chunk.groupby(['user_id', 'behavior_type']).size().reset_index(name='count')
            user_chunks.append(user_agg)

            # 2. Item Dimension
            item_agg = chunk.groupby(['item_id', 'behavior_type']).size().reset_index(name='count')
            item_chunks.append(item_agg)

            # 3. Category Dimension
            cat_agg = chunk.groupby(['item_category', 'behavior_type']).size().reset_index(name='count')
            category_chunks.append(cat_agg)

            # 4. Interaction Dimension (User-Item Pair)
            inter_agg = chunk.groupby(['user_id', 'item_id', 'behavior_type']).size().reset_index(name='count')
            interaction_chunks.append(inter_agg)

            logging.info("Processed intermediate chunk...")

        logging.info("All chunks read. Performing final aggregations and calculating ratios...")

        # Process and save User Features
        user_features = pivot_and_rename(pd.concat(user_chunks), ['user_id'])
        user_features.to_csv(config['user_features_file'], index=False, encoding='utf-8-sig')
        logging.info(f"User features saved to '{config['user_features_file']}'")

        # Process and save Item Features
        item_features = pivot_and_rename(pd.concat(item_chunks), ['item_id'])
        item_features.to_csv(config['item_features_file'], index=False, encoding='utf-8-sig')
        logging.info(f"Item features saved to '{config['item_features_file']}'")

        # Process and save Category Features
        category_features = pivot_and_rename(pd.concat(category_chunks), ['item_category'])
        category_features.to_csv(config['category_features_file'], index=False, encoding='utf-8-sig')
        logging.info(f"Category features saved to '{config['category_features_file']}'")

        # Process and save Interaction Features
        interaction_features = pivot_and_rename(pd.concat(interaction_chunks), ['user_id', 'item_id'])
        interaction_features.to_csv(config['user_item_features_file'], index=False, encoding='utf-8-sig')
        logging.info(f"Interaction features saved to '{config['user_item_features_file']}'")

    except Exception as e:
        logging.error(f"An error occurred during feature extraction: {e}")
        raise


if __name__ == "__main__":
    extract_features()