import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def predict_probabilities():
    try:
        # 1. Load the trained LightGBM model
        model = joblib.load("../data/processed/taobao_behavior_model.pkl")

        # 2. Load the master data to look up the user and item features
        df = pd.read_csv("../data/processed/master_training_data.csv")

        # Define the exact same columns we dropped during training to prevent leakage
        leakage_columns = [
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "total_actions",
            "buy_rate",
        ]
        columns_to_drop = ["user_id", "item_id", "target_label"] + leakage_columns

        print("\n=== Taobao Recommendation Engine ===")
        # Get a random sample to test
        sample_row = df.sample(1).iloc[0]
        test_user = int(sample_row["user_id"])
        test_item = int(sample_row["item_id"])

        print(f"Testing User ID: {test_user}")
        print(f"Testing Item ID: {test_item}")
        print("-" * 35)

        # Extract just this row's features and format it for the model
        feature_vector = sample_row.drop(labels=columns_to_drop).to_frame().T

        # 3. Predict the exact probabilities for all 4 classes
        probabilities = model.predict_proba(feature_vector)[0]

        # 4. Display the results clearly
        print("Predicted User Intent Probabilities:")
        print(f" View (Type 1):     {probabilities[0] * 100:>6.2f}%")
        print(f"  Favorite (Type 2): {probabilities[1] * 100:>6.2f}%")
        print(f"Cart (Type 3):     {probabilities[2] * 100:>6.2f}%")
        print(f"Buy (Type 4):      {probabilities[3] * 100:>6.2f}%")
        print("===================================\n")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    predict_probabilities()
